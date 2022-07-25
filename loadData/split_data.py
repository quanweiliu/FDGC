import torch
import numpy as np

class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, patch_size=5, flip_augmentation=False, radiation_augmentation=False, mixture_augmentation=False, remove_zero_labels=True):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
            mixture_augmentation  不能用
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.ignored_labels = set()
        self.center_pixel = True
        self.flip_augmentation = flip_augmentation
        self.radiation_augmentation = radiation_augmentation
        self.mixture_augmentation = mixture_augmentation
        self.remove_zero_labels = remove_zero_labels
    
        # print(supervision)
        mask = np.ones_like(gt)
        # print("mask", mask.shape) 
        
        # 说是非零的索引，因为是新创建的 ones_like matrix，所以是返回的所有位置的索引 
        x_pos, y_pos = np.nonzero(mask)                         # Return the indices of the elements that are non-zero.
        # print("x_pos", x_pos.shape, "y_pos", y_pos.shape) 
        p = self.patch_size // 2

        # 为什么把最外围的像素都给删除了？ 我选择不删除
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                # if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
                if x >= p and x < data.shape[0] - p and y >= p and y < data.shape[1] - p
            ]
        )
        # print("self.indices", self.indices.shape)                # (21025, 2)
        # print("self.indices 0", self.indices)                # (21025, 2)

        self.labels = [self.label[x, y] for x, y in self.indices]
        # print("self.labels", len(self.labels))                   # 21025
        # print("self.label 0", self.labels)

        # remove zero labels
        if self.remove_zero_labels:
            self.indices = np.array(self.indices)
            self.labels = np.array(self.labels)
            # print("type 1", type(self.indices))
            # print("self.indices 1", self.indices)                # (21025, 2)
            # print("self.label 1", self.labels)
            

            self.indices = self.indices[self.labels>0]
            self.labels = self.labels[self.labels>0]
            # print("type 2", type(self.indices))
            # print("self.indices 2", self.indices)                # (21025, 2)
            # print("self.label 2", self.labels)

            # self.indices = [list(i) for i in self.indices]
            # self.labels = [list(i) for i in self.labels]

    # 三种数据增强策略
    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    # mixture_noise 不能用，出错。
    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        '''
            x, y -> index
            x1, y1 = x - 4, y - 4
            x2, y2 = x, y
        '''
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # 三种数据集增强
        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
            
        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        return data, label


def sample_gt(gt, train_num=50, train_ratio=0.1, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    # print("test_gt", test_gt.shape)

    if mode == 'number':
        print("split_type: ", mode, "\ntrain_number: ", train_num)
        sample_num = train_num
        for c in np.unique(gt):
            if c == 0:
              continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices)) 
            y = gt[indices].ravel()  
            np.random.shuffle(X)

            max_index = np.max(len(y)) + 1
            if sample_num > max_index:
                sample_num = 15
            else:
                sample_num = train_num

            train_indices = X[: sample_num]
            test_indices = X[sample_num:]

            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]

            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'ratio':
        print("split_type: ", mode, "\ntrain_ratio: ", train_ratio)
        for c in np.unique(gt):
            if c == 0:
              continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices)) 
            y = gt[indices].ravel()   
            np.random.shuffle(X)

            train_num = np.ceil(train_ratio * len(y)).astype('int')
            # print(train_num)

            train_indices = X[: train_num]
            test_indices = X[train_num:]
            
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            # print("test_indices", test_indices)

            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                # numpy.count_nonzero是用于统计数组中非零元素的个数
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_ratio:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0
        test_gt[train_gt > 0] = 0

    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))

    return train_gt, test_gt






















































