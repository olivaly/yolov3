import torch
import numpy as np
import cv2
import os
import shutil
from pathlib import Path

import torchvision.ops


# bbox (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax)
def xywh2x1y1x2y2(bbox):
    bbox_ = bbox.new(bbox.shape)
    bbox_[..., 0] = bbox[..., 0] - bbox[..., 2] / 2
    bbox_[..., 2] = bbox[..., 0] + bbox[..., 2] / 2
    bbox_[..., 1] = bbox[..., 1] - bbox[..., 3] / 2
    bbox_[..., 3] = bbox[..., 1] + bbox[..., 3] / 2
    return bbox_


# ---------------- 读取类别名称及个数 ------------------
def get_classes(data_path):
    with open(data_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if line.split(':')[0] == 'names':
                info = line.split(':')[-1]
                info = info.replace('\'', '').replace('[', '').replace(']', '').split(',')
    return len(info), info

# ---------------------- 解析label文件 ----------------------
def read_labels(label_path, image_size, h, w, x1y1x2y2=False):
    values = []
    for i in range(len(label_path)):
        value = []
        with open(label_path[i], 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line != '\n':
                    value.append(line.replace('\n', '').split(' '))
        if x1y1x2y2:
            value_ = torch.zeros(len(value), 5)
            for i, v in enumerate(value):
                value_[i, 4] = float(v[0])
                value_[i, 0] = float(v[1]) - float(v[3]) / 2
                value_[i, 1] = float(v[2]) - float(v[4]) / 2
                value_[i, 2] = float(v[1]) + float(v[3]) / 2
                value_[i, 3] = float(v[2]) + float(v[4]) / 2
        else:
            value_ = torch.tensor([list(map(float, v)) for v in value])

        if h != w:
            if w > h:
                padw, padh = 0, (w - h) // 2
            elif w < h:
                padw, padh = (h - w) // 2, 0

            temps = []
            for v in value_:
                temp = [v[0]]
                temp.extend((v[1] * w + padw) / image_size)
                temp.extend((v[2] * h + padh) / image_size)
                temp.extend(v[3] * (image_size / w))
                temp.extend(v[4] * (image_size / h))
                temps.append(temp)
            value_ = temps
        values.append(value_)
        # labels = convert_labels(values)
    return values

# --------------- 网络模型参数初始化 -----------------
def weight_init(model, init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(classname, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

# --------------------- 输出结果解码 --------------------
def decode_pred(preds, input_size, anchors, anchor_mask, num_classes):
    """
    :param preds: 通过模型推理得到的输出值，三个尺度：
                    batch_size, 255, 13, 13
                    batch_size, 255, 26, 26
                    batch_size, 255, 52, 52

    将输出的尺度w, h划分为网格 ----> grid_x, grid_y
    x = sigmoid(pred[..., 0]) + grid_x
    y = sigmoid(pred[..., 1]) + grid_y
    w =  exp(pred[..., 2]) * anchor_w
    h = exp(pred[..., 3]) * anchor_h
    conf = sigmoid(pred[..., 4])
    cls = sigmoid(pred[..., 5:])
    :return: (batch_size, 3*height*weight, 5+num_class)
    """
    outputs = []
    for i, pred in enumerate(preds):
        batch_size = pred.size(0)
        input_height = pred.size(2)
        input_weight = pred.size(3)

        stride_w, stride_h = input_size / input_weight, input_size / input_height
        scaled_anchor = [(anchors[i][0]/stride_w, anchors[i][1]/stride_h) for i in anchor_mask[i]]

        pred = pred.view(batch_size, len(anchor_mask[i]), num_classes+5, input_weight, input_height).permute(0, 1, 3, 4, 2).contiguous()
        pred = pred.detach().cpu()
        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]
        conf = torch.sigmoid(pred[..., 4])
        cls = torch.sigmoid(pred[..., 5:])

        grid_x = torch.linspace(0, input_weight-1, input_weight).repeat(input_height, 1).repeat(batch_size*len(anchor_mask[i]), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, input_height-1, input_height).repeat(input_weight, 1).t().repeat(batch_size*len(anchor_mask[i]), 1, 1).view(y.shape).type_as(y)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        anchor_w = FloatTensor(scaled_anchor).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchor).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_weight*input_height).view(w.shape).type_as(w)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_weight*input_height).view(h.shape).type_as(h)

        pred_boxes = FloatTensor(pred[..., :4].shape)
        pred_boxes[..., 0] = grid_x + x.data
        pred_boxes[..., 1] = grid_y + y.data
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # 将输出结果归一化
        scale_ = FloatTensor([input_weight, input_height, input_weight, input_height])
        pred_boxes = pred_boxes.view(batch_size, -1, 4) / scale_
        output = torch.cat((pred_boxes, conf.view(batch_size, -1, 1), cls.view(batch_size, -1, num_classes)), -1)
        outputs.append(output.data)
    return outputs
def non_max_suppression(preds, conf_thres=0.45, nms_thres=0.25, max_nms=3000, c=70000):
    bs = preds.shape[0]
    output = [torch.zeros((0, 6),device=preds.device)] * bs
    box = xywh2x1y1x2y2(preds[..., :4])
    preds[..., :4] = box

    for i, pred in enumerate(preds):
        # 按置信度进行筛选
        pred = pred[pred[:, 4] >= conf_thres]
        cls_conf, cls_index = torch.max(pred[:, 5:], 1, keepdim=True)
        dets = torch.cat((pred[:, :4], cls_conf.float()*pred[:, 4:5], cls_index.float()), 1)[cls_conf.view(-1)*pred[:, 4] >= conf_thres]

        if not dets.shape[0]:
            continue

        boxes, score = dets[:, :4]+c, dets[:, 5]
        keep_index = torchvision.ops.nms(boxes, score, nms_thres)
        keep_index = keep_index[:max_nms]
        max_det = dets[keep_index]
        output[i] = max_det if output[i] is None else torch.cat((output[i], max_det))
        output[i] = output[i].detach().cpu()
    return output

def increment_path(path, exist_ok=False, sep=""):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    return path


# ---------------------------- 画出检测框 ------------------------------
COLOR = [(255,125,0),(255,255,0),(255,0,125),(255,0,250),(255,125,125),
         (255,125,250),(125,125,0),(0,255,125),(255,0,0), (255,0,0),
         (0,0,255),(125,0,255),(0,125,255),(0,255,255),(125,125,255),
         (0,255,0),(125,255,125),(255,255,255),(100,100,100),(0,0,0)]  # 用来标识20个类别的bbox颜色，可自行设定

def draw_bbox(img_path, bboxes1, input_size, classes, save_path):
    img = cv2.imread(img_path)
    _, img_name = os.path.split(img_path)
    img = cv2.resize(img, (input_size, input_size))
    h, w = img.shape[0:2]
    for i in range(len(bboxes1)):
        if np.sum(bboxes1[i] == None):
            continue
        for j in range(bboxes1[i].shape[0]):
            p1 = (max(int(w * bboxes1[i][j, 0]), 0), max(int(h * bboxes1[i][j, 1]), 0))
            p2 = (max(int(w * bboxes1[i][j, 2]), 0), max(int(h * bboxes1[i][j, 3]), 0))
            class_name = classes[int(bboxes1[i][j, 5])]
            cv2.rectangle(img, p1, p2, COLOR[int(bboxes1[i][j, 5])], 2)
            cv2.putText(img, class_name+' '+str(np.round(bboxes1[i][j, 4].item(), decimals=2)), (p1[0], p1[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR[int(bboxes1[i][j, 5])], 1)
    cv2.imwrite(os.path.join(save_path, img_name), img)