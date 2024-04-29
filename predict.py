import argparse
import os, stat

import numpy as np
from torchvision.transforms import transforms
import torch
from PIL import Image
from utils import *
from model import YoloModel
import shutil
import re

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--image_size', type=int, default=416)
    parser.add_argument("--device", default='0', help='cuda device, 0, 1, 2, 3 or cpu')
    parser.add_argument("--conf_threshold", type=int, default=0.3, help='confidence threshold')
    parser.add_argument("--nms_threshold", type=int, default=0.25)
    parser.add_argument("--anchor_mask", type=list, default=[[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    parser.add_argument("--anchors", type=list, default=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119],
                                                         [116, 90], [156, 198], [373, 326]])
    parser.add_argument("--model_path", type=str, default=r'E:\经典模型复现\Detect\yolov3\checkpoints2\last.pt',
                        help='model weights save path')
    parser.add_argument("--dataset_dir", type=str, default=r"E:\经典模型复现\Dataset\neu_det_datasets\images\val", help='images to be inferenced')
    parser.add_argument("--save_path", type=str, default=r"E:\经典模型复现\Detect\yolov3\result", help='save predict results')
    parser.add_argument('--label_path', type=str, default=r"E:\经典模型复现\Dataset\neu_det_datasets\labels\val")
    # parser.add_argument('--label_save_path', type=str, default=r"E:\经典模型复现\Detect\yolov3\result\label_imgs")
    parser.add_argument("--data", type=str, default=r"E:\经典模型复现\Dataset\neu_det_datasets\dataset.yaml")
    return parser.parse_args()

def read_labels(label_path):
    fr = open(label_path, 'r', encoding='utf8')
    lines = fr.readlines()
    labels = []
    for line in lines:
        line = line.replace('\n', '')
        line = list(map(float, re.findall(r"\d+\.?\d*", line)))
        label = np.zeros((1, 6), dtype=np.float32)
        label[:, 0] = line[1] - line[3] / 2
        label[:, 1] = line[2] - line[4] / 2
        label[:, 2] = line[1] + line[3] / 2
        label[:, 3] = line[2] + line[4] / 2
        label[:, 4] = 1.0
        label[:, 5] = line[0]
        labels.append(label)
    return labels

def test(opts):
    print('============================ Start Inference ============================')
    img_list = os.listdir(opts.dataset_dir)
    trans = transforms.Compose([
        transforms.Resize([opts.image_size, opts.image_size]),
        transforms.ToTensor()
    ])
    num_classes, classes = get_classes(opts.data)

    # ------------------ load model ------------------
    model = YoloModel(opts.anchor_mask, num_classes)
    ckpt = torch.load(opts.model_path)
    model.load_state_dict(ckpt['model_weights'])
    model.eval()

    if opts.device != 'cpu':
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.device
    else:
        device = 'cpu'
    model.to(device)

    save_path = increment_path(Path(opts.save_path) / 'infer_result')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # if opts.label_save_path:
    #     if os.path.exists(opts.label_save_path):
    #         shutil.rmtree(opts.label_save_path)
    #     os.makedirs(opts.label_save_path)

    for i, img_name in enumerate(img_list):
        img_path = os.path.join(opts.dataset_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_ = trans(img)
        img_ = torch.unsqueeze(img_, dim=0)
        img_ = img_.to(device)
        preds = model(img_)
        output = decode_pred(preds, opts.image_size, opts.anchors, opts.anchor_mask, num_classes)
        bboxes = torch.cat(output, 1)  # 3 * (batch_size, _, num_classes+5)
        bboxes1 = non_max_suppression(bboxes, conf_thres=opts.conf_threshold, nms_thres=opts.nms_threshold)
        draw_bbox(img_path, bboxes1, opts.image_size,  classes, save_path)

        # label_path = os.path.join(opts.label_path, img_name.split('.')[0]+'.txt')
        # labels = read_labels(label_path)
        # draw_bbox(img_path, labels, opts.image_size, classes, opts.label_save_path)
    print(" Inference finished! ")

if __name__=='__main__':
    opts = parse_opt()
    test(opts)
