import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import os
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.util import tensor_tonemap
from basicsr.models.vgg import Vgg16
import numpy as np
import Imath, OpenEXR
import cv2

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class IOException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class TestImageEventRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(TestImageEventRestorationModel, self).__init__(opt)

        # define network
        self.prev_states = None
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        self.pixel_type = train_opt['loss'].pop('type')
        # print('LOSS: pixel_type:{}'.format(self.pixel_type))
        self.criterionL1 = torch.nn.L1Loss().to(self.device)
        self.vgg = Vgg16(requires_grad=False).to(self.device)
        self.lambda_L1 = train_opt['loss'].pop('lambda_L1')
        self.lambda_perc = train_opt['loss'].pop('lambda_perc')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                [{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                **train_opt['optim_g'])
        # elif optim_type == 'SGD':
        #     self.optimizer_g = torch.optim.SGD(optim_params,
        #                                        **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        # print(self.optimizer_g)
        # exit(0)

    def feed_data(self, input):

        self.input_ldr_rgbs = input['ldr_rgb'].to(self.device)
        self.input_evs = input['ev'].to(self.device)
        self.gt_hdr_rgbs = input['gt_hdr_rgb'].to(self.device)
        self.image_paths = input['paths']

    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def grids_voxel(self):
        b, c, h, w = self.voxel.size()
        self.original_size_voxel = self.voxel.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)

        # print('step_i, stepj', step_i, step_j)
        # exit(0)

        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.voxel[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.voxel[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})

        self.origin_voxel = self.voxel
        self.voxel = torch.cat(parts, dim=0)
        print('----------parts voxel .. ', len(parts), self.voxel.size())
        self.idxes = idxes

    def grids(self):
        b, c, h, w = self.lq.size()  # lq is after data augment (for example, crop, if have)
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)

        # print('step_i, stepj', step_i, step_j)
        # exit(0)

        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h - crop_size)
                j = random.randint(0, w - crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        print('...', self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(
                self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq
        self.voxel = self.origin_voxel

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # preds = self.net_g(self.lq)
        if self.opt['datasets']['train'].get('use_mask'):
            # print('NETWORK TRAIN USE MASK')
            # print('MASK.SHAPE:{}'.format(self.mask.shape))
            # print('MASK:{}'.format(self.mask))
            preds = self.net_g(x=self.lq, event=self.voxel, mask=self.mask)

        else:
            preds = self.net_g(x=self.lq, event=self.voxel)

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.

            if self.pixel_type == 'PSNRATLoss':
                l_pix += self.cri_pix(*preds, self.gt)

            elif self.pixel_type == 'PSNRGateLoss':
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt, self.mask)

            elif self.pixel_type == 'PSNRLoss':
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        # if self.cri_perceptual:
        #
        #
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():

            n, t = self.input_evs.shape[:2]
            self.gt = self.gt_hdr_rgbs[:, t - 1]
            self.event = self.input_evs[:, t - 1]
            self.ldr = self.input_ldr_rgbs[:, t - 1]
            self.path = self.image_paths[t - 1][0]
            # 获取当前视频的编号
            # current_video_id = self.path.split('\\')[-2]  # 从路径中提取视频文件夹编号

            # 如果路径格式不同，可以根据实际情况修改这行
            current_video_id = os.path.basename(os.path.dirname(self.path))  # 更通用的方式
            # # 或者
            # current_video_id = self.path.split('/')[-2]  # 如果使用正斜杠分隔

            # 如果是新的视频序列，重置 prev_states
            if not hasattr(self, 'last_video_id') or self.last_video_id != current_video_id:
                self.prev_states = None
                self.last_video_id = current_video_id
                self.t_im = None
                self.t_gt = None
            # n = self.ldr.size(0)  # n: batch size
            m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n

                output = []
                pre, self.prev_states = self.net_g(x=self.ldr, event=self.event, prev_states=self.prev_states)
                # self.prev_states = [[tensor.detach() for tensor in state_list] for state_list in states]
                output.append(pre)
                i = j

            # self.prev_states = None
            self.output = torch.cat(output, dim=0)  # all mini batch cat in dim0
            # self.output_tm = tensor_tonemap(self.output)
            # self.gt_tm = tensor_tonemap(self.gt)

            # self.output = torch.clamp(self.output, 0, 1)
            self.tm_output, self.t_im = tonemap(self.output, self.t_im)
            # self.tm_output = torch.clamp(self.tm_output, 0, 1)
            # self.output = torch.clamp(self.output, 0, 1)
            self.tm_gt, self.t_gt = tonemap(self.gt, self.t_gt)
            # self.gt = torch.clamp(self.gt, 0, 1)

        self.net_g.train()

    def single_image_inference(self, img, voxel, save_path):
        self.feed_data(data={'frame': img.unsqueeze(dim=0), 'voxel': voxel.unsqueeze(dim=0)})

        if self.opt['val'].get('grids') is not None:
            self.grids()
            self.grids_voxel()

        self.test()

        if self.opt['val'].get('grids') is not None:
            self.grids_inverse()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, save_path)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        logger = get_root_logger()
        # logger.info('Only support single GPU validation.')
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = self.opt.get('name')

        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = os.path.splitext(os.path.basename(val_data['paths'][0][0]))[0]
            # img_name = '{:04d}'.format(cnt)
            self.feed_data(val_data)
            if self.opt['val'].get('grids') is not None:
                self.grids()
                self.grids_voxel()

            self.test()

            if self.opt['val'].get('grids') is not None:
                self.grids_inverse()

            visuals = self.get_current_visuals()

            if save_img:

                if self.opt['is_train'] == False:  # TRAIN
                    print('Save path:{}'.format(self.opt['path']['visualization']))
                    print('Dataset name:{}'.format(dataset_name))
                    print('Img_name:{}'.format(img_name))
                    # ldr_path = osp.join(
                    #     self.opt['path']['visualization'], dataset_name, self.last_video_id,
                    #     f'{img_name}_ldr.png')
                    # ldr_numpy = tensor2im(visuals['ldr'])
                    # imwrite(ldr_numpy, ldr_path)

                    gt_tm_path = osp.join(
                        self.opt['path']['visualization'], self.last_video_id,
                        f'{img_name}_gt_tm.png')
                    gt_tm_numpy = tensor2im(visuals['gt_tm'])
                    imwrite(gt_tm_numpy, gt_tm_path)

                    result_path = osp.join(
                        self.opt['path']['visualization'], self.last_video_id,
                        f'{img_name}_result.hdr')
                    result_numpy = tensor2im(visuals['result'], imtype=np.float64)
                    write_hdr(result_numpy.astype(np.float32), result_path)
                    # writeEXR(result_numpy, result_path)

                    result_tm_path = osp.join(
                        self.opt['path']['visualization'], self.last_video_id,
                        f'{img_name}_result.png')
                    result_tm_numpy = tensor2im(visuals['result_tm'])
                    imwrite(result_tm_numpy, result_tm_path)
                    # if 'gt' in visuals:
                    # if 'event' in visuals:
                    # image_numpy = make_event_preview(image[:, i:i + 3, :, :])
                    # img_path = os.path.join(savedir, '%s_%s.jpg' % (name, label))
                    # cv2.imwrite(img_path, image_numpy)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            cnt += 1
            # if cnt == 300:
            #     break
        pbar.close()

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        # out_dict['ldr'] = self.ldr.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['result_tm'] = self.tm_output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
            out_dict['gt_tm'] = self.tm_gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


def tonemap(img, log_sum_prev=None):
    key_fac, epsilon, tm_gamma = 0.5, 1e-6, 1.4
    XYZ = BGR2XYZ(img)
    b, c, h, w = XYZ.shape
    if log_sum_prev is None:
        log_sum_prev = torch.log(epsilon + XYZ[:, 0, :, :]).sum((1, 2), keepdim=True)
        log_avg_cur = torch.exp(log_sum_prev / (h * w))
        key = key_fac
    else:
        log_sum_cur = torch.log(XYZ[:, 1, :, :] + epsilon).sum((1, 2), keepdim=True)
        log_avg_cur = torch.exp(log_sum_cur / (h * w))
        log_avg_temp = torch.exp((log_sum_cur + log_sum_prev) / (2.0 * h * w))
        key = key_fac * log_avg_cur / log_avg_temp
        log_sum_prev = log_sum_cur
    Y = XYZ[:, 1, :, :]
    Y = Y / log_avg_cur * key
    Lmax = torch.max(torch.max(Y, 1, keepdim=True)[0], 2, keepdim=True)[0]
    L_white2 = Lmax * Lmax
    L = Y * (1 + Y / L_white2) / (1 + Y)
    XYZ *= (L / XYZ[:, 1, :, :]).unsqueeze(1)
    image = XYZ2BGR(XYZ)
    image = torch.clamp(image, 0, 1) ** (1 / tm_gamma)
    return image, log_sum_prev


def BGR2XYZ(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    b = image[..., 0, :, :]
    g = image[..., 1, :, :]
    r = image[..., 2, :, :]

    X = (0.4124 * r) + (0.3576 * g) + (0.1805 * b)
    Y = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
    Z = (0.0193 * r) + (0.1192 * g) + (0.9505 * b)

    return torch.stack((X, Y, Z), -3)


def XYZ2BGR(image):
    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    X = image[..., 0, :, :]
    Y = image[..., 1, :, :]
    Z = image[..., 2, :, :]

    r = (3.240625 * X) + (-1.537208 * Y) + (-0.498629 * Z)
    g = (-0.968931 * X) + (1.875756 * Y) + (0.041518 * Z)
    b = (0.055710 * X) + (-0.204021 * Y) + (1.056996 * Z)

    return torch.stack((b, g, r), -3)


def tensor2im(input_image, imtype=np.uint8):
    """"
    Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (1, 2, 0))

        # from [-1,1] to [0,1]
        # image_numpy = ((image_numpy+1.0)/2.0)*255  # post-processing: tranpose and scaling
        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
        image_numpy = image_numpy * 255

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image[0]

    return image_numpy.astype(imtype)


def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        B = (img[:, :, 0]).astype(np.float16).tobytes()  # 通过将 tostring() 替换为 tobytes()
        G = (img[:, :, 1]).astype(np.float16).tobytes()
        R = (img[:, :, 2]).astype(np.float16).tobytes()
        out.writePixels({'R': R, 'G': G, 'B': B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s" % e)


def write_hdr(hdr_image, path):
    """ Writing HDR image in radiance (.hdr) format """

    norm_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    with open(path, "wb") as f:
        norm_image = (norm_image - norm_image.min()) / (
                norm_image.max() - norm_image.min()
        )  # normalisation function
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (norm_image.shape[0], norm_image.shape[1]))
        brightest = np.maximum(
            np.maximum(norm_image[..., 0], norm_image[..., 1]), norm_image[..., 2]
        )
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((norm_image.shape[0], norm_image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(norm_image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)
        rgbe.flatten().tofile(f)
        f.close()