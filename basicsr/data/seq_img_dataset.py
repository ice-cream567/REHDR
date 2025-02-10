# 按照HDRev的加载方式，voxel[-1,1] ldr[-1,1] hdr[0,*)

import os.path
import torchvision.transforms as transforms
import numpy as np
from basicsr.data.image_folder import make_dataset
from PIL import Image
import cv2
from basicsr.utils.util import readEXR, writeEXR, whiteBalance
from torch.utils import data as data
from basicsr.data.h5_augment import *
from torch.utils.data import ConcatDataset
import pandas as pd
import glob


def concatenate_datasets(dataset_class, opt):
    """连接多个子文件夹中的数据集"""
    root_path = opt['dataroot']
    dir_ldr = os.path.join(opt['dataroot'], 'LDR/')
    dir_hdr = os.path.join(opt['dataroot'], 'HDR/')
    dir_event = os.path.join(opt['dataroot'], 'event/')
    # 检查必要的文件夹是否存在
    if not all(os.path.isdir(d) for d in [dir_ldr, dir_hdr, dir_event]):
        raise Exception(f'Missing required directories in {root_path}')

    # 获取LDR文件夹下的所有子文件夹作为基准
    subfolders = sorted([d for d in os.listdir(dir_ldr)
                         if os.path.isdir(os.path.join(dir_ldr, d))])

    if not subfolders:
        raise Exception(f'No subfolders found in {dir_ldr}')

    print(f'Found {len(subfolders)} subfolders: {subfolders}')

    # 检查每个子文件夹在三个目录下都存在
    datasets = []
    for subfolder in subfolders:
        ldr_subfolder = os.path.join(dir_ldr, subfolder)
        hdr_subfolder = os.path.join(dir_hdr, subfolder)
        event_subfolder = os.path.join(dir_event, subfolder)

        # 验证对应的子文件夹都存在
        if not all(os.path.isdir(d) for d in [ldr_subfolder, hdr_subfolder, event_subfolder]):
            print(f'Warning: Skipping {subfolder} due to missing corresponding folders')
            continue

        # 创建数据集实例
        subfolder_paths = {
            'ldr': ldr_subfolder,
            'hdr': hdr_subfolder,
            'event': event_subfolder
        }
        datasets.append(dataset_class(opt, subfolder_paths))

    if not datasets:
        raise Exception('No valid datasets could be created')

    print(f'Successfully created {len(datasets)} datasets')
    return ConcatDataset(datasets)


class SeqImgDataset(data.Dataset):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #
    #     return parser

    def __init__(self, opt, subfolder_paths, return_voxel=True, return_ldr=True, return_hdr=True, norm_voxel=True):
        super(SeqImgDataset, self).__init__()
        self.opt = opt

        self.dir_ldr = subfolder_paths['ldr']
        self.dir_hdr = subfolder_paths['hdr']
        self.dir_voxel = subfolder_paths['event']

        self.ldr_paths = sorted(make_dataset(self.dir_ldr, opt['max_dataset_size']))
        self.hdr_paths = sorted(make_dataset(self.dir_hdr, opt['max_dataset_size']))
        self.voxel_paths = sorted(make_dataset(self.dir_voxel, opt['max_dataset_size']))
        self.ldr_size = len(self.ldr_paths)

        self.return_format = 'torch'
        # self.return_voxel = return_voxel
        self.return_ldr = return_ldr
        self.return_hdr = opt.get('return_hdr', return_hdr)
        self.return_voxel = opt.get('return_voxel', return_voxel)

        self.norm_voxel = norm_voxel  # -MAX~MAX -> -1 ~ 1
        self.transforms = {}
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        if self.opt['norm_voxel'] is not None:
            self.norm_voxel = self.opt['norm_voxel']  # -MAX~MAX -> -1 ~ 1

        if self.opt['return_voxel'] is not None:
            self.return_voxel = self.opt['return_voxel']

        if self.opt['crop_size'] is not None:
            self.transforms["RandomCrop"] = {"size": self.opt['crop_size']}

        if self.opt['use_flip']:
            self.transforms["RandomFlip"] = {}

        if 'LegacyNorm' in self.transforms.keys() and 'RobustNorm' in self.transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in self.transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in self.transforms.items()]
                del (self.transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

        transforms_list = [eval(t)(**kwargs) for t, kwargs in self.transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)

        if not self.normalize_voxels:
            self.vox_transform = self.transform

    def __getitem__(self, index, seed=None):
        if index < 0 or index >= self.__len__():
            raise IndexError
        seed = random.randint(0, 2 ** 32) if seed is None else seed
        data_ldr = []
        data_hdr = []
        data_ev = []
        ldr_paths = []

        for i in range(index, index + self.opt['time_step']):
            ldr_path = self.ldr_paths[i]
            hdr_path = self.hdr_paths[i]
            voxel_path = self.voxel_paths[i]

            # ----------- Load HDR Image -------------
            if hdr_path[-4:] == '.hdr':
                hdr_img = cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH)
                hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
            else:
                hdr_img = readEXR(hdr_path)
                # if hdr_img.min() < 0:
                #     hdr_img = hdr_img - hdr_img.min()
            hdr_img = (hdr_img - hdr_img.min()) / (hdr_img.max() - hdr_img.min())

            # ----------- Load LDR Image -------------
            ldr_img = Image.open(ldr_path).convert('RGB')
            ldr_img = np.array(ldr_img)
            ldr_img = ldr_img / 255.0
            ldr_img = np.power(ldr_img, 2.2)

            voxel = np.load(voxel_path).astype(np.float32)

            # apply image transformation
            hdr_img = self.transform_frame(hdr_img, seed, transpose_to_CHW=True)
            ldr_img = self.transform_frame(ldr_img, seed, transpose_to_CHW=True)
            # ldr_norm = ldr_img * 2.0 - 1.0
            voxel = self.transform_voxel(voxel, seed, transpose_to_CHW=False)

            data_ldr.append(ldr_img)
            data_hdr.append(hdr_img)
            data_ev.append(voxel)
            ldr_paths.append(ldr_path)
        return {'ev': torch.stack(data_ev),
                'gt_hdr_rgb': torch.stack(data_hdr),
                'ldr_rgb': torch.stack(data_ldr),
                'paths': ldr_paths
                }

    def __len__(self):
        return self.ldr_size

    def transform_frame(self, frame, seed, transpose_to_CHW=False):
        """
        Augment frame and turn into tensor
        @param frame Input frame
        @param seed  Seed for random number generation
        @returns Augmented frame
        """
        if self.return_format == "torch":
            if transpose_to_CHW:
                frame = torch.from_numpy(frame.transpose(2, 0, 1)).float()  # H,W,C -> C,H,W

            else:
                frame = torch.from_numpy(frame).float()  # 0-1
            if self.transform:
                random.seed(seed)
                frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed, transpose_to_CHW):
        """
        Augment voxel and turn into tensor
        @param voxel Input voxel
        @param seed  Seed for random number generation
        @returns Augmented voxel
        """
        if self.return_format == "torch":
            if transpose_to_CHW:
                voxel = torch.from_numpy(voxel.transpose(2, 0, 1)).float()  # H,W,C -> C,H,W

            else:
                if self.norm_voxel:
                    voxel = torch.from_numpy(voxel).float() / abs(max(voxel.min(), voxel.max(), key=abs))  # -1 ~ 1
                else:
                    voxel = torch.from_numpy(voxel).float()


            if self.vox_transform:
                random.seed(seed)
                voxel = self.vox_transform(voxel)
        return voxel


    # def transform_voxel(self, voxel, seed, transpose_to_CHW):
    #     """
    #     Augment voxel and turn into tensor
    #     @param voxel Input voxel
    #     @param seed  Seed for random number generation
    #     @returns Augmented voxel
    #     """
    #     if self.return_format == "torch":
    #         # 首先转换为torch tensor
    #         if transpose_to_CHW:
    #             voxel = torch.from_numpy(voxel.transpose(2, 0, 1)).float()  # H,W,C -> C,H,W
    #         else:
    #             voxel = torch.from_numpy(voxel).float()
    #
    #         # 应用正则化
    #         if self.norm_voxel:
    #             # 找到非零元素
    #             nonzero_mask = (voxel != 0)
    #             num_nonzeros = nonzero_mask.sum()
    #
    #             if num_nonzeros > 0:
    #                 # 计算非零元素的均值和标准差
    #                 mean = voxel[nonzero_mask].mean()
    #                 stddev = voxel[nonzero_mask].std()
    #
    #                 # 只对非零元素进行标准化
    #                 voxel[nonzero_mask] = (voxel[nonzero_mask] - mean) / stddev
    #
    #         # 应用额外的体素变换（如果有的话）
    #         if self.vox_transform:
    #             random.seed(seed)
    #             voxel = self.vox_transform(voxel)
    #
    #     return voxel

# def transform_voxel(self, voxel, seed, transpose_to_CHW):
#     """
#     Augment voxel and turn into tensor
#     @param voxel Input voxel
#     @param seed  Seed for random number generation
#     @returns Augmented voxel
#     """
#     if self.return_format == "torch":
#         # 首先转换为torch tensor
#         if transpose_to_CHW:
#             voxel = torch.from_numpy(voxel.transpose(2, 0, 1)).float()  # H,W,C -> C,H,W
#         else:
#             voxel = torch.from_numpy(voxel).float()
#
#         # 应用正则化
#         if self.norm_voxel:
#             if hasattr(self, 'norm_type') and self.norm_type == 'z_score':
#                 # Z-score标准化（只处理非零值）
#                 nonzero_mask = (voxel != 0)
#                 num_nonzeros = nonzero_mask.sum()
#
#                 if num_nonzeros > 0:
#                     mean = voxel[nonzero_mask].mean()
#                     stddev = voxel[nonzero_mask].std()
#                     voxel[nonzero_mask] = (voxel[nonzero_mask] - mean) / stddev
#             else:
#                 # 原始的Min-Max归一化到[-1,1]
#                 voxel = voxel / abs(max(voxel.min(), voxel.max(), key=abs))
#
#         # 应用额外的体素变换（如果有的话）
#         if self.vox_transform:
#             random.seed(seed)
#             voxel = self.vox_transform(voxel)
#
#     return voxel