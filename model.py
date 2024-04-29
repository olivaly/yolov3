import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, inplanes):
        super(Conv, self).__init__()
        kernel_size = inplanes[2]
        pad = (kernel_size - 1) // 2 if kernel_size else 0
        self.conv = nn.Conv2d(inplanes[0], inplanes[1], kernel_size=inplanes[2], stride=inplanes[3], padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(inplanes[1])
        self.relu1 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu1(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, inputs):
        super(ResBlock, self).__init__()
        self.Conv1 = Conv([inputs[1], inputs[0], 1, 1])
        self.Conv2 = Conv([inputs[0], inputs[1], 3, 1])

    def forward(self, x):
        residual = x
        y = self.Conv1(x)
        y = self.Conv2(y)
        y += residual
        return y

# ------------ BackBone --------------
class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.Conv = Conv([3, 32, 3, 1])
        self.layer1 = self.make_layers([32, 64], 1)
        self.layer2 = self.make_layers([64, 128], 2)
        self.layer3 = self.make_layers([128, 256], 8)
        self.layer4 = self.make_layers([256, 512], 8)
        self.layer5 = self.make_layers([512, 1024], 4)

        self.layers_out_filters = [64, 128, 256, 512, 1024]

    def make_layers(self, inputs, blocks):
        layer = [Conv([inputs[0], inputs[1], 3, 2])]
        for i in range(blocks):
            layer += [ResBlock([inputs[0], inputs[1]])]
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.Conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)       # (10, 256, 52, 52)
        out4 = self.layer4(out3)    # (10, 512, 26, 26)
        out5 = self.layer5(out4)    # (10, 1024, 13, 13)
        return out3, out4, out5

# --------------  head --------------
class YoloModel(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrain=False):
        super(YoloModel, self).__init__()
        self.backbone = DarkNet53()
        if pretrain:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        out_filter = self.backbone.layers_out_filters
        self.last_layer0 = self.YoloNeck([512, 1024], out_filter[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_conv1 = Conv([512, 256, 1, 1])
        self.last_upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = self.YoloNeck([256, 512], out_filter[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_conv2 = Conv([256, 128, 1, 1])
        self.last_upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = self.YoloNeck([128, 256], out_filter[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def YoloNeck(self, filter_list, inplane, outplane):
        neck = nn.Sequential(
                Conv([inplane, filter_list[0], 1, 1]),
                Conv([filter_list[0], filter_list[1], 3, 1]),
                Conv([filter_list[1], filter_list[0], 1, 1]),
                Conv([filter_list[0], filter_list[1], 3, 1]),
                Conv([filter_list[1], filter_list[0], 1, 1]),
                Conv([filter_list[0], filter_list[1], 3, 1]),
                nn.Conv2d(filter_list[1], outplane, kernel_size=1, stride=1, padding=0, bias=True)
        )
        return neck

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)

        out0_branch = self.last_layer0[:5](x0)
        out0 = self.last_layer0[5:](out0_branch)

        x1_in = self.last_conv1(out0_branch)
        x1_in = self.last_upsample1(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)

        out1_branch = self.last_layer1[:5](x1_in)
        out1 = self.last_layer1[5:](out1_branch)

        x2_in = self.last_conv2(out1_branch)
        x2_in = self.last_upsample2(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)

        out2 = self.last_layer2(x2_in)
        return out0, out1, out2


if __name__=='__main__':
    input = torch.zeros(10, 3, 416,416)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 6
    net = YoloModel(anchor_mask, num_classes)
    print(net)
    out = net(input)
    print('out0:', out[0].shape)   # batch_size, len(anchor_mask[0]) * (num_class + 5）,  13, 13
    print('out1:', out[1].shape)   # batch_size, len(anchor_mask[1]) * (num_class + 5）,  26, 26
    print('out2:', out[2].shape)   # batch_size, len(anchor_mask[2]) * (num_class + 5）,  52, 52
