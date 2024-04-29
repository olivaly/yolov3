import torch
import math
import numpy as np
from utils import xywh2x1y1x2y2

# 通过K-means算法在训练集中所有样本的真实框中聚类得到几个不同尺寸的框（加入尺寸先验经验）
"""
13 * 13特征图上的感受野最大， anchor设置为 116*90， 156*198， 373*326，适合于对大目标的检测
26 * 26具有中等感受野， anchor设置为 30*61， 62*45， 59*119
52 * 52最大特征图上具有最小的感受野，10*13， 16*30， 33*23适合于小目标的检测
1.确定真实框在哪个cell
2.由这个cell的三个anchor与真实框计算IOU,选择IOU最大的一个框
3.该框通过平移（tx, ty)和尺度缩放（tw,th)微调，使anchor box与ground truth重合
"""

class Compute_Loss:
    def __init__(self, anchors, num_classes, input_shape, device, anchor_mask, get_giou=False):
        super(Compute_Loss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.anchor_mask = anchor_mask

        self.get_giou = get_giou
        self.ignore_thresh = 0.5
        self.coord_ratio = 0.05
        self.obj_ratio = 5.0
        self.cls_ratio = 1 * (num_classes / 80)
        self.balance = [0.4, 1, 4]
        if device == 'cuda':
            self.cuda = True
        else:
            self.cuda = False

    def forward(self, l, pred, target):
        bs, in_h, in_w = pred.size(0), pred.size(2), pred.size(3)   # bacthsize, h, w

        stride_h, stride_w = self.input_shape / in_h, self.input_shape / in_w
        scale_anchor = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors] # 对anchor进行尺度缩放
        # 对真实框的参数进行编码
        y_true, noobj_mask, bbox_scale_loss = self.get_target(l, target, scale_anchor, in_h, in_w)

        pred_ = pred.view(bs, len(self.anchor_mask[l]), (5+self.num_classes), in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        # 预测框的参数
        x = torch.sigmoid(pred_[..., 0])
        y = torch.sigmoid(pred_[..., 1])
        w = pred_[..., 2]
        h = pred_[..., 3]
        conf = torch.sigmoid(pred_[..., 4])
        cls = torch.sigmoid(pred_[..., 5:])

        # 对预测框进行解码
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, w, h, target, scale_anchor, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true = y_true.type_as(x)
            noobj_mask = noobj_mask.type_as(x)
            bbox_scale_loss = bbox_scale_loss.type_as(x)

        bbox_scale_loss = 2 - bbox_scale_loss
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)
        loss = 0
        if n != 0:
            # 仅对正样本进行loss_loc和loss_cls的计算
            if self.get_giou:
                giou = self.calculate_giou(pred_boxes, y_true[..., :4]).type_as(x)
                loss_loc = torch.mean((1 - giou)[obj_mask])
            else:
                loss_x = torch.mean(self.BCELoss(x[obj_mask], y_true[..., 0][obj_mask]) * bbox_scale_loss[obj_mask])
                loss_y = torch.mean(self.BCELoss(y[obj_mask], y_true[..., 1][obj_mask]) * bbox_scale_loss[obj_mask])
                loss_w = torch.mean(self.MSELoss(w[obj_mask], y_true[..., 2][obj_mask]) * bbox_scale_loss[obj_mask])
                loss_h = torch.mean(self.MSELoss(h[obj_mask], y_true[..., 3][obj_mask]) * bbox_scale_loss[obj_mask])
                loss_loc = (loss_x + loss_y + loss_w + loss_h) * 0.1
            loss_cls = torch.mean(self.BCELoss(cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.coord_ratio
            loss += loss_cls * self.cls_ratio
        loss_conf = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss
    def get_target(self, l, target, scale_anchor, in_h, in_w):
        """
        :param l: anchor_mask的层数
        :param target: 真实框
        :param scale_anchor: 归一化后的先验anchor box
        :param h: pred的高
        :param w: pred的宽
        :return: y_true：将target编码成网格偏移量的形式,
                 noobj_mask：标记和target没有交集的值，
                 bbox__loss_scale：target缩放到pred的缩放尺度
        先找到与target最接近的anchor box，再根据真实框和anchor box反推偏移值
        """
        bs = len(target) # batch_size
        noobj_mask = torch.ones(bs, len(self.anchor_mask[l]), in_h, in_w) # 不包含物体的先验框
        bbox_loss_scale = torch.zeros(bs, len(self.anchor_mask[l]), in_h, in_w)
        y_true = torch.zeros(bs, len(self.anchor_mask[l]), in_h, in_w, 5+self.num_classes)

        for b in range(bs):
            if len(target[b]) == 0:
                 continue
            batch_target = torch.zeros_like(target[b])
            batch_target[:, [0, 2]] = target[b][:, [1, 3]] * in_w
            batch_target[:, [1, 3]] = target[b][:, [2, 4]] * in_h
            batch_target[:, 4] = target[b][:, 0]
            batch_target = batch_target.cpu()

            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))

            anchor_shape = torch.FloatTensor(torch.cat((torch.zeros((len(scale_anchor), 2)), torch.FloatTensor(scale_anchor)), 1))

            ious = self.calculate_iou(gt_box, anchor_shape)
            best_ns = torch.argmax(ious, dim=-1)  # 选取iou最大的anchor

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchor_mask[l]:
                    continue
                k = self.anchor_mask[l].index(best_n)  # 确定是哪一个先验框
                i, j = torch.floor(batch_target[t, 0]).long(), torch.floor(batch_target[t, 1]).long() # 确定先验框中心点在哪个cell
                c = batch_target[t, 4].long() # 真实框的class
                # 检测框有目标点， noobj_mask置0
                noobj_mask[b, k, j, i] = 0
                if not self.get_giou:
                    y_true[b, k, j, i, 0] = batch_target[t, 0] - i  # 真实框的中心点偏移dx
                    y_true[b, k, j, i, 1] = batch_target[t, 1] - j  # 真实框的中心点偏移dy
                    y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / scale_anchor[best_n][0])  # 真实框的大小偏移tw
                    y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / scale_anchor[best_n][1])  # 真实框的大小偏移th
                    y_true[b, k, j, i, 4] = 1  # 真实框的置信度
                    y_true[b, k, j, i, 5+c] = 1  # 真实框的类别
                else:
                    y_true[b, k, j, i, 0] = batch_target[t, 0]
                    y_true[b, k, j, i, 1] = batch_target[t, 1]
                    y_true[b, k, j, i, 2] = batch_target[t, 2]
                    y_true[b, k, j, i, 3] = batch_target[t, 3]
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, 5+c] = 1
                bbox_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / (in_w * in_h) # 面积尺度归一化比例
        return y_true, noobj_mask, bbox_loss_scale

    def get_ignore(self, l, x, y, w, h, target, scale_anchor, in_h, in_w, noobj_mask):
        """
        :param l: 第l层anchor_mask
        :param x: pred的中心横坐标偏移
        :param y: pred的中心纵坐标偏移
        :param w: pred的宽偏移量
        :param h: pred的高偏移量
        :param target: 真实框
        :param scale_anchor: pred尺度归一化后的anchor
        :param noobj_mask: 标记no objection的mask
        通过比对所有网格cell和anchor bbox再加上预测的偏移量后哪些和target的iou(重叠度）最高
        :return: noobj_mask: 与target的iou大于ignore_threshold的anchor bbox在noobj_mask上标记为0
                 pred_bbox: 预测框的具体参数值（xywh)
        """
        bs = len(target)
        # 生成网格， 先验框的中心
        grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_h, 1).repeat(int(len(self.anchor_mask[l])*bs), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_w, 1).t().repeat(int(len(self.anchor_mask[l])*bs), 1, 1).view(y.shape).type_as(y)

        # 生成先验框的宽高
        anchor_l = np.array(scale_anchor)[self.anchor_mask[l]]
        anchor_w = torch.tensor(anchor_l).index_select(1, torch.tensor(0)).type_as(x)
        anchor_h = torch.tensor(anchor_l).index_select(1, torch.tensor(1)).type_as(x)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h*in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h*in_w).view(h.shape)

        # 根据偏移量计算调整后的预测框的中心与宽高
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(anchor_w * torch.exp(w), -1)
        pred_boxes_h = torch.unsqueeze(anchor_h * torch.exp(h), -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(bs):
            pred_boxes_ = pred_boxes[b].view(-1, 4)
            if len(target[b]):
                batch_target = torch.zeros_like(target[b])
                batch_target[:, [0, 2]] = target[b][:, [1, 3]] * in_w
                batch_target[:, [1, 3]] = target[b][:, [2, 4]] * in_h
                batch_target = batch_target.type_as(x)
                anch_ious = self.calculate_iou(batch_target[..., :4], pred_boxes_)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_thresh] = 0
        return noobj_mask, pred_boxes


    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target, eps=1e-7):
        pred = self.clip_by_tensor(pred, eps, 1-eps)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output
    def calculate_iou(self, bbox1, bbox2, x1y1x2y2=True):
        """
        计算bbox1和bbox2的iou或giou
        bbox: xywh
        """
        if x1y1x2y2:
            bbox1 = xywh2x1y1x2y2(bbox1)
            bbox2 = xywh2x1y1x2y2(bbox2)

        bbox1_, bbox2_ = bbox1.unsqueeze(1).repeat(1, len(bbox2), 1), bbox2.unsqueeze(0).repeat(len(bbox1), 1, 1)
        intersect_bbox = torch.zeros(len(bbox1), len(bbox2), 4).type_as(bbox1)  # bbox1和bbox2的重合区域的(x1,y1,x2,y2)

        intersect_bbox[:, :, 0] = torch.max(bbox1_[:, :, 0], bbox2_[:, :, 0])
        intersect_bbox[:, :, 1] = torch.max(bbox1_[:, :, 1], bbox2_[:, :, 1])
        intersect_bbox[:, :, 2] = torch.min(bbox1_[:, :, 2], bbox2_[:, :, 2])
        intersect_bbox[:, :, 3] = torch.min(bbox1_[:, :, 3], bbox2_[:, :, 3])

        w = torch.clamp(intersect_bbox[:, :, 2] - intersect_bbox[:, :, 0], min=0)
        h = torch.clamp(intersect_bbox[:, :, 3] - intersect_bbox[:, :, 1], min=0)
        area1 = (bbox1_[:, :, 2] - bbox1_[:, :, 0]) * (bbox1_[:, :, 3] - bbox1_[:, :, 1])  # bbox1面积
        area2 = (bbox2_[:, :, 2] - bbox2_[:, :, 0]) * (bbox2_[:, :, 3] - bbox2_[:, :, 1])  # bbox2面积
        intersect_area = w * h # 交集面积
        iou = intersect_area.type_as(bbox1) / (area1 + area2 - intersect_area + 1e-6)  # 防止除0
        return iou

    def calculate_giou(self, bbox1, bbox2):
        """
        :param bbox1:  tensor, shape = (batch, feat_w, feat_h, anchor_num, 4), xywh
        :param bbox2:  tensor, shape = (batch, feat_w, feat_h, anchor_num, 4), xywh
        :return: giou, tensor, shape = (batch, feat_w, feat_h, anchor_num, 4), xywh
         giou = iou - abs(C-AUB)/abs(C)
        """

        bbox1, bbox2 = xywh2x1y1x2y2(bbox1), xywh2x1y1x2y2(bbox2)
        x_min, y_min = torch.max(bbox1[..., 0], bbox2[..., 0]), torch.max(bbox1[..., 1], bbox2[..., 1])
        x_max, y_max = torch.min(bbox1[..., 2], bbox2[..., 2]), torch.min(bbox1[..., 3], bbox2[..., 3])
        w = torch.max(x_max - x_min, torch.zeros_like(x_max))
        h = torch.max(y_max - y_min, torch.zeros_like(y_max))

        area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
        area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])
        intersect_area = w * h
        union_area = area1 + area2 - intersect_area
        iou = intersect_area / (union_area + 1e-6)

        enclose_xmin, enclose_ymin = torch.min(bbox1[..., 0], bbox2[..., 0]), torch.min(bbox1[..., 1], bbox2[..., 1])
        enclose_xmax, enclose_ymax = torch.max(bbox1[..., 2], bbox2[..., 2]), torch.max(bbox1[..., 3], bbox2[..., 3])
        enclose_w = torch.max(enclose_xmax - enclose_xmin, torch.zeros_like(x_max))
        enclose_h = torch.max(enclose_ymax - enclose_ymin, torch.zeros_like(y_max))
        enclose_area = enclose_h * enclose_w
        giou = iou - (enclose_area - union_area) / enclose_area
        return giou


if __name__ == '__main__':
    # -----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
    #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
    #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
    # -----------------------------------------------------------#
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    num_classes = 6
    input_shape = 416
    device = 'cuda'
    loss = Compute_Loss(anchors, num_classes,input_shape, device, anchor_mask=[[6,7,8], [3,4,5], [0,1,2]])
    pred = torch.rand((16, 3*(num_classes+5), 13, 13))
    target = [torch.rand(1, 5) for i in range(16)]
    loss = loss.forward(0, pred, target)
    print('loss:', loss)