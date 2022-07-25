import torch
import numpy as np 
import random
import yaml
from loadData import data_reader
from loadData.split_data import HyperX, sample_gt

# only active in this file
def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def get_data(model_name="FDGC", 
    path_config=None, 
    print_config=False, print_data_info=False):
    
    config = yaml.load(open(path_config, "r"), Loader=yaml.FullLoader)
    # config

    dataset_name = config["data_input"]["dataset_name"]
    path_data = config["data_input"]["path_data"]
    patch_size = config["data_input"]["patch_size"]

    split_type = config["data_split"]["split_type"]
    train_num = config["data_split"]["train_num"]
    val_num = config["data_split"]["val_num"]
    train_ratio = config["data_split"]["train_ratio"]
    val_ratio = config["data_split"]["val_ratio"]

    num_components = config["data_transforms"]["num_components"]
    batch_size = config["data_transforms"]["batch_size"]
    remove_zero_labels = config["data_transforms"]["remove_zero_labels"]
    start = config["result_output"]["data_info_start"]

    data, data_gt = data_reader.load_data(dataset_name, path_data=path_data)
    # split_data.data_info(data_gt, class_num=np.max(data_gt))

    data, pca = data_reader.apply_PCA(data, num_components=num_components)
    pad_width = 19 // 2
    img = np.pad(data, pad_width=pad_width, mode="constant", constant_values=(0))        # 111104
    img = img[:, :, pad_width:img.shape[2]-pad_width]
    gt = np.pad(data_gt, pad_width=pad_width, mode="constant", constant_values=(0))
    # print(img.shape, gt.shape)

    train_gt, test_gt = sample_gt(gt, train_num=train_num, train_ratio=train_ratio, mode=split_type)
    # data_reader.data_info(train_gt, test_gt)

    # obtain label
    train_label, test_label = [], []
    for i in range(pad_width, train_gt.shape[0]-pad_width):
        for j in range(pad_width, train_gt.shape[1]-pad_width):
            if train_gt[i][j]:
                train_label.append(train_gt[i][j])

    for i in range(pad_width, test_gt.shape[0]-pad_width):
        for j in range(pad_width, test_gt.shape[1]-pad_width):
            if test_gt[i][j]:
                test_label.append(test_gt[i][j])
    # len(test_label)
    
    # print control
    if print_config:
        print(config)
    if print_data_info:
        data_reader.data_info(train_gt, test_gt, start=start)


    # create dataloader
    train_dataset = HyperX(img, train_gt, patch_size=patch_size, flip_augmentation=False, 
                            radiation_augmentation=False, mixture_augmentation=False, 
                            remove_zero_labels=remove_zero_labels)
    test_dataset = HyperX(img, test_gt, patch_size=patch_size, flip_augmentation=False, 
                            radiation_augmentation=False, mixture_augmentation=False, 
                            remove_zero_labels=remove_zero_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False        )

    return train_loader, test_loader, train_label, test_label
















