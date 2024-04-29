from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import argparse

class ToYoloDataset(Dataset):
    def __init__(self, opts, mode='train', trans=None):
        self.image_size = opts.image_size
        self.data_path = opts.dataset_dir
        if mode == 'class_names':
            self.classes = read_dataset_path(self.data_path, 'names')
        elif mode == 'train' or mode == 'val':
            image_path = read_dataset_path(self.data_path, mode)  # 解析数据存储路径
            self.images_list, self.labels_list = self.load_data(image_path)  # 加载数据并将其转换为yolov1训练格式数据
        self.trans = trans
    def get_classes(self):
        self.classes = self.classes.replace('\'', '').replace('[', '').replace(']', '').split(',')
        return self.classes, len(self.classes)

    # ----- 加载数据并修改数据格式 -----
    def load_data(self, images_path):
        images_list, labels_list = [], []
        labels_path = images_path.replace('images', 'labels')
        labels_dir, images_dir = os.listdir(labels_path), os.listdir(images_path)
        assert len(labels_dir) == len(images_dir), '-- labels files are not equal images.'

        for image_index in sorted(images_dir):
            img = cv2.imread(os.path.join(images_path, image_index))
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            if h != w:
                padding_img(img)
            if max(h, w) != self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size))
            images_list.append(img)
            label_index = image_index.split('.')[0] + '.txt'
            # labels = self.read_labels(os.path.join(labels_path, label_index), h, w)
            labels_list.append(os.path.join(labels_path, label_index))
        return images_list, labels_list

    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, item):
        label = self.labels_list[item]
        image = self.images_list[item]
        if self.trans is None:
            trans = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
        else:
            trans = self.trans
        image = trans(image)
        return image, label


# ----- 加载yaml文件 ------
def read_dataset_path(data_path, mode):
    with open(data_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if line.split(':')[0] == mode:
                info = line.split(':')[-1]
    return info

# 转换标签格式为tensor: (batch_size, 6) [x, y, w, h, index, cls_index]
def convert_labels_format(labels):
    n = sum([i.shape[0] for i in labels])
    targets = torch.zeros(n, 6)
    st = 0
    for i in range(len(labels)):
        step = labels[i].shape[0]
        targets[st:st+step, 0] = i
        targets[st:st+step, 1:] = labels[i]
        st += step
    return targets

def padding_img(img):
    # padding image into square
    h, w = img.size()
    pad = abs(h - w) // 2
    img = np.pad(img, ((0,0), (pad, pad), (0,0)), 'constant', constant_values=128)
    return img


if __name__ == '__main__':
    # 调试用，依次取出数据看看是否正确
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=416)
    parser.add_argument('--dataset_dir', type=str, default=r"E:\经典模型复现\Dataset\neu_det_datasets\dataset.yaml")
    opts = parser.parse_args()
    classes, num_class = ToYoloDataset(opts, mode='class_names').get_classes()
    dataset = ToYoloDataset(opts, mode='train')
    dataloader = DataLoader(dataset, 1)
    print('classes:', classes)
    print('numbers of classes:', num_class)
    for i in enumerate(dataloader):
        input("press enter to continue")