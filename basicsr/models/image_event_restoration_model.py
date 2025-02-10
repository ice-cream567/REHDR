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

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    normalized = (batch - mean) / std
    return normalized


def Gram_matrix(input):
    a, b, c, d = input.size()
    features = input.reshape(a * b, c * d)
    G = torch.mm(features, features.t())/(a * b * c * d)

    return G

class ImageEventRestorationModel(BaseModel):
    """Base Event-based deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageEventRestorationModel, self).__init__(opt)

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
        self.lambda_L1=train_opt['loss'].pop('lambda_L1')
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
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                [{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                **train_opt['optim_g'])

        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    # def feed_data(self, data):
    #
    #     self.lq = data['frame'].to(self.device)
    #     self.voxel=data['voxel'].to(self.device)
    #     if 'mask' in data:
    #         self.mask = data['mask'].to(self.device)
    #     if 'frame_gt' in data:
    #         self.gt = data['frame_gt'].to(self.device)
    def set_input(self, input):
        # self.input_ldr_ys = input['input_ldr_y']
        self.input_ldr_rgbs = input['ldr_rgb'].to(self.device)
        self.input_evs = input['ev'].to(self.device)
        # if (self.is_train == True):
            # self.gt_hdr_ys = input['gt_hdr_y']
            # self.gt_hdr_uvs = input['gt_hdr_uv']
        self.gt_hdr_rgbs = input['gt_hdr_rgb'].to(self.device)
        # self.input_ldr_us = input['input_ldr_u']
        # self.input_ldr_vs = input['input_ldr_v']
        self.image_paths = input['paths']

        # for i in range(len(self.input_ldr_ys)):
        #     self.input_ldr_ys[i] = self.input_ldr_ys[i].to(self.device)
        #     self.input_ldr_rgbs[i] = self.input_ldr_rgbs[i].to(self.device)
        #     self.input_evs[i] = self.input_evs[i].to(self.device)
        #     self.gt_hdr_ys[i] = self.gt_hdr_ys[i].to(self.device)
        #     self.gt_hdr_uvs[i] = self.gt_hdr_uvs[i].to(self.device)
        #     self.gt_hdr_rgbs[i] = self.gt_hdr_rgbs[i].to(self.device)
        #     self.input_ldr_us[i] = self.input_ldr_us[i].to(self.device)
        #     self.input_ldr_vs[i] = self.input_ldr_vs[i].to(self.device)

    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def backward_G_seq(self):
        # 使用 torch.zeros 初始化损失，并移动到正确的设备上
        loss_G_L1 = torch.zeros(1, device=self.outputs[0].device)
        loss_G_perc = torch.zeros(1, device=self.outputs[0].device)

        for i in range(len(self.outputs)):
            # Compute L1 loss
            tmp_gt = tensor_tonemap(self.gts[i])
            tmp_output = tensor_tonemap(self.outputs[i])

            # 使用普通加法而不是原地加法
            l1_loss = self.criterionL1(tmp_output, tmp_gt) * self.lambda_L1
            loss_G_L1 = loss_G_L1 + l1_loss

            # Compute perceptual loss on colored hdr
            output_hdr_features = self.vgg(normalize_batch(tmp_output))
            gt_hdr_features = self.vgg(normalize_batch(tmp_gt))

            # 初始化特征损失
            feat_loss = torch.zeros(1, device=self.outputs[0].device)

            for f_x, f_y in zip(output_hdr_features, gt_hdr_features):
                mse_loss = torch.mean((f_x - f_y) ** 2)
                G_x = Gram_matrix(f_x)
                G_y = Gram_matrix(f_y)
                gram_loss = torch.mean((G_x - G_y) ** 2)
                feat_loss = feat_loss + mse_loss + gram_loss

            loss_G_perc = loss_G_perc + feat_loss * self.lambda_perc

        # 存储中间损失值
        self.loss_G_L1 = loss_G_L1
        self.loss_G_perc = loss_G_perc

        # 计算总损失
        if self.pixel_type == 'l1+perc':
            self.loss_G = (loss_G_L1 + loss_G_perc) / len(self.gts)
        else:
            self.loss_G = loss_G_L1 / len(self.gts)

        # 反向传播
        self.loss_G.backward()

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
        # self.loss_output_hdr_rgbs = []
        # self.loss_gt_hdr_rgbs = []
        # self.output_hdr_rgbs = []
        self.outputs = []
        self.gts = []
        B, T = self.input_evs.shape[:2]
        self.prev_states = None
        for t in range(T):
            self.event = self.input_evs[:,t]
            self.ldr = self.input_ldr_rgbs[:,t]
            pre, states = self.net_g(x=self.ldr, event=self.event, prev_states=self.prev_states)
            self.prev_states = [[tensor.detach() for tensor in state_list] for state_list in states]
            self.outputs.append(pre)
            self.gts.append(self.gt_hdr_rgbs[:,t])
        self.prev_states = None

        self.backward_G_seq()
        # use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        # if use_grad_clip:
        #     torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        loss_dict = OrderedDict()
        loss_dict[self.pixel_type] = self.loss_G
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        self.net_g.eval()
        with torch.no_grad():

            n, t = self.input_evs.shape[:2]
            self.gt = self.gt_hdr_rgbs[:, t-1]
            self.event = self.input_evs[:, t-1]
            self.ldr = self.input_ldr_rgbs[:, t-1]
            self.path = self.image_paths[t-1][0]
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
            self.output_tm = tensor_tonemap(self.output)
            self.gt_tm = tensor_tonemap(self.gt)
        self.net_g.train()

    def single_image_inference(self, img, voxel, save_path):
        self.feed_data(data={'frame': img.unsqueeze(dim=0), 'voxel': voxel.unsqueeze(dim=0)})
        if self.opt['val'].get('grids') is not None:
            self.grids()
            self.grids_voxel()

        self.test()

        if self.opt['val'].get('grids') is not None:
            self.grids_inverse()
            # self.grids_inverse_voxel()

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
        dataset_name = self.opt.get('name')  # !

        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = '{:08d}'.format(cnt)

            self.set_input(val_data)
            if self.opt['val'].get('grids') is not None:
                self.grids()
                self.grids_voxel()

            self.test()

            if self.opt['val'].get('grids') is not None:
                self.grids_inverse()

            visuals = self.get_current_visuals()
            # tm_result = tensor_tonemap(visuals['result'])
            # tm_gt = tensor_tonemap(visuals['gt'])
            sr_img = tensor2img([visuals['result_tm']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt_tm']], rgb2bgr=rgb2bgr)
                del self.gt_tm

            # tentative for out of GPU memory
            del self.ldr
            del self.output_tm
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:
                    if cnt == 1:  # visualize cnt=1 image every time
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                    img_name,
                                                    f'{img_name}_{current_iter}_gt.png')
                else:
                    print('Save path:{}'.format(self.opt['path']['visualization']))
                    print('Dataset name:{}'.format(dataset_name))
                    print('Img_name:{}'.format(img_name))
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            cnt += 1
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
        out_dict['ldr'] = self.ldr.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['result_tm'] = self.output_tm.detach().cpu()
        if hasattr(self, 'gts' or 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
            out_dict['gt_tm'] = self.gt_tm.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
