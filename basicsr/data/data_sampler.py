import math
import torch
from torch.utils.data.sampler import Sampler
#
#
# class EnlargedSampler(Sampler):
#     """Sampler that restricts data loading to a subset of the dataset.
#
#     Modified from torch.utils.data.distributed.DistributedSampler
#     Support enlarging the dataset for iteration-based training, for saving
#     time when
#
#     restart the dataloader after each epoch
#
#     Args:
#         dataset (torch.utils.data.Dataset): Dataset used for sampling.
#         num_replicas (int | None): Number of processes participating in
#             the training. It is usually the world_size.
#         rank (int | None): Rank of the current process within num_replicas.
#         ratio (int): Enlarging ratio. Default: 1.
#     """
#
#     def __init__(self, dataset, num_replicas, rank, ratio=1, time_step=10): #!
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.time_step = time_step#!
#         self.effective_size = len(range(0, len(dataset), time_step))#!
#         # 根据ratio计算采样数量
#         self.num_samples = math.ceil(
#             self.effective_size * ratio / self.num_replicas)
#         # self.num_samples = math.ceil(
#         #     len(self.dataset) * ratio / self.num_replicas)
#         self.total_size = self.num_samples * self.num_replicas
#
#     def __iter__(self):
#         # deterministically shuffle based on epoch
#         g = torch.Generator()
#         g.manual_seed(self.epoch)
#         # indices = torch.randperm(self.total_size, generator=g).tolist()
#         # 首先生成考虑时间步长的索引
#         base_indices = list(range(0, len(self.dataset), self.time_step))
#         # 如果需要扩充数据集
#         if self.total_size > len(base_indices):
#             # 计算需要重复的次数
#             num_repeats = math.ceil(self.total_size / len(base_indices))
#             # 重复基础索引
#             base_indices = base_indices * num_repeats
#             # 截断到所需大小
#             base_indices = base_indices[:self.total_size]
#
#         # 打乱索引
#         indices = torch.tensor(base_indices)[torch.randperm(len(base_indices), generator=g)].tolist()
#
#
#         # dataset_size = len(self.dataset)
#         # indices = [v % dataset_size for v in indices]
#
#         # subsample
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         assert len(indices) == self.num_samples
#
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_samples
#
#     def set_epoch(self, epoch):
#         self.epoch = epoch

class EnlargedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, ratio=1, time_step=10):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.time_step = time_step  # 连续帧数量

        # 处理ConcatDataset
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            self.cumulative_sizes = dataset.cumulative_sizes
            self.dataset_lengths = []
            for ds in dataset.datasets:
                self.dataset_lengths.append(len(ds))
        else:
            self.dataset_lengths = [len(dataset)]
            self.cumulative_sizes = self.dataset_lengths.copy()

        # 计算每个数据集的有效索引，确保每个索引后面至少有 time_step-1 帧
        self.valid_indices = []
        current_position = 0

        for i, dataset_len in enumerate(self.dataset_lengths):
            # 只使用能够获取连续 time_step 帧的索引
            max_valid_idx = dataset_len - (time_step - 1)  # 最后的有效索引
            if max_valid_idx <= 0:
                continue  # 如果数据集太小，跳过

            dataset_indices = list(range(0, max_valid_idx))

            # 如果是ConcatDataset，需要调整索引以匹配ConcatDataset的索引机制
            if isinstance(dataset, torch.utils.data.ConcatDataset):
                if i > 0:
                    offset = self.cumulative_sizes[i - 1]
                else:
                    offset = 0
                dataset_indices = [idx + offset for idx in dataset_indices]

            self.valid_indices.extend(dataset_indices)

        self.effective_size = len(self.valid_indices)

        # 计算采样数量
        self.num_samples = math.ceil(
            self.effective_size * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        if self.total_size > self.effective_size:
            # 计算需要重复的次数
            num_repeats = math.ceil(self.total_size / self.effective_size)
            # 重复基础索引
            indices = self.valid_indices * num_repeats
            # 截断到所需大小
            indices = indices[:self.total_size]
        else:
            indices = self.valid_indices.copy()

        # 打乱索引
        indices = torch.tensor(indices)[torch.randperm(len(indices), generator=g)].tolist()

        # subsample for distributed training
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

