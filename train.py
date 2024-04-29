import os
import datetime
import time
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import YoloModel

from data import ToYoloDataset
from loss import Compute_Loss
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import weight_init,read_labels, get_classes

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default=None, help="initial weights path")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.00125, help="learning rate")
    parser.add_argument('--image_size', type=int, default=416)  # 图片输入大小必须为32的倍数
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--checkpoints_dir", type=str, default='./checkpoints3', help='model weights save path')
    parser.add_argument("--device", default=0, help='cuda device, 0, 1, 2, 3 or cpu(-1)')
    parser.add_argument('--num_workers', type=int, default=8, help="use n threads to read data")
    parser.add_argument("--dataset_dir", type=str, default=r"E:\经典模型复现\Dataset\neu_det_datasets\dataset.yaml")
    parser.add_argument('--save_freq', type=int, default=10, help='model save frequency')
    parser.add_argument("--anchor_mask", type=list, default=[[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    parser.add_argument("--anchors", type=list, default=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119],
                                                        [116, 90], [156, 198], [373, 326]])
    return parser.parse_args()

def Trainer(opts):
    print('============================ Preparing Training Settings ============================')
    if os.path.exists(opts.checkpoints_dir):
        shutil.rmtree(opts.checkpoints_dir)
    os.mkdir(opts.checkpoints_dir)
    show_config(pretrain=opts.pretrain, epochs=opts.epochs, batch_size=opts.batch_size, learning_rate=opts.lr,
                image_size=opts.image_size, weight_decay=opts.weight_decay, weight_save_path=opts.checkpoints_dir,
                device=opts.device, num_workers=opts.num_workers, dataset_dir=opts.dataset_dir, save_frequency=opts.save_freq)
    # ---------------------- 加载数据 ------------------------
    num_classes, classes = get_classes(opts.dataset_dir)
    train_dataset = ToYoloDataset(opts, mode='train', trans=None)
    val_dataset = ToYoloDataset(opts, mode='val', trans=None)
    train_loader = DataLoader(train_dataset, opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=opts.num_workers)
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    # ----------------------- 加载模型 -----------------------
    model = YoloModel(opts.anchor_mask, num_classes)


    if opts.pretrain is not None:
        print('Load weights {}.'.format(opts.pretrain))
        model = torch.load(opts.pretrain)
    else:
        weight_init(model)

    if opts.device != -1:
        device = 'cuda'
        model.to(device)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.device)
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        model.to(device)

    # ----------------------- 加载优化器及损失函数 ------------------------
    compute_loss = Compute_Loss(opts.anchors, num_classes, opts.image_size, device, anchor_mask=opts.anchor_mask)
    optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.937, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-4)

    print("============================ Start training ============================")
    train_losses = []
    best_metric = 1000000
    t0 = time.time()
    for e in range(opts.epochs):
        # training
        t1 = time.time()
        train_loss = train(model, train_loader, optimizer, compute_loss, e, num_train, opts)
        t2 = time.time()
        train_losses.append(train_loss)

        ckpt = {
            'epoch': e,
            'model': model,
            'model_weights': model.state_dict(),
            'training time': (t2 - t1),
            'date': time.time(),
            'best_metric': best_metric,
        }
        save_model(ckpt, opts, 'last.pt')
        # validation
        val_metrics = validate(model, val_loader, compute_loss, e, num_val, opts)
        if val_metrics < best_metric:
            best_metric = val_metrics
            ckpt = {
                'epoch': e,
                'model': model,
                'model_weights': model.state_dict(),
                'training time': (t2 - t1),
                'date': time.time(),
                'best_metric': best_metric,
            }

        print("Epoch %d is now the best epoch with metric %.4f\n" % (e+1, best_metric))
        with open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+") as log_file:
            log_file.write("Training consumes %.2f second\n" % (t2-t1))
            log_file.write("Epoch %d is now the best epoch with metric %.4f\n" % (e, best_metric))
        if e % opts.save_freq == 0 or e == opts.epochs+1:
            save_model(ckpt, opts, 'best.pt')
        draw_loss_curve(train_losses, opts)
    te = time.time()
    print('total training time is %.2f hours', (te-t0)/3600)

#训练
def train(model, train_loader, optimizer, compute_loss, epoch, num_train, opts):
    model.train()
    losssum = 0.

    if opts.device != -1:
        device = 'cuda'
    else:
        device = 'cpu'

    # log_file是保存网络训练过程信息的文件，网络训练信息会以追加的形式打印在log.txt里，不会覆盖原有log文件
    log_file = open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+")
    localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 打印训练时间
    log_file.write(localtime)
    log_file.write("\n====================== training epoch %d ======================\n"%epoch)
    pbar = tqdm(total=num_train//opts.batch_size, desc=f'Epoch {epoch + 1}/{opts.epochs}', postfix=dict, mininterval=0.3)
    for i, (imgs, label_path) in enumerate(train_loader):
        imgs = imgs.to(device)
        h, w = imgs.shape[-2:]
        labels = read_labels(label_path, opts.image_size, h, w)
        # labels = convert_labels_format(labels)
        preds = model(imgs)  # 前向传播
        loss = 0
         # 计算损失
        for l in range(len(preds)):
            loss += compute_loss.forward(l, preds[l], labels)
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 优化网络参数
        losssum += loss.item()
        pbar.set_postfix(**{'loss': losssum / (i + 1)})
        pbar.update(1)
    log_file.write("Epoch %d/%d | training loss = %.3f\n" %
              (epoch+1, opts.epochs, losssum/(len(train_loader))))
    log_file.flush()
    log_file.close()
    return losssum / num_train

def validate(model, val_loader, compute_loss, epoch, num_val, opts):
    model.eval()
    log_file = open(os.path.join(opts.checkpoints_dir, "log.txt"), "a+")
    log_file.write("====================== validate epoch %d ======================\n"%epoch)
    sum_loss = 0.  # 计算损失
    with torch.no_grad():  # 加上这个可以减少在validation过程时的显存占用，提高代码的显存利用率
        # pbar = tqdm(enumerate(val_loader), total=num_val)
        for i,(imgs, label_path) in enumerate(val_loader):
            if opts.device != 'cpu':
                device = 'cuda'
            loss = 0.
            imgs = imgs.to(device)
            h, w = imgs.shape[-2:]
            labels = read_labels(label_path, opts.image_size, h, w)
            # labels = convert_labels_format(labels)
            preds = model(imgs)  # 前向传播
            for l in range(len(preds)):
                loss += compute_loss.forward(l, preds[l], labels)
            # pbar.set_postfix(**{'val_loss': loss/(i+1)})
            # pbar.update(1)
            sum_loss += loss
        avg_loss = sum_loss/len(val_loader)
        print('Evaluation of validation: average loss = %.5f'% avg_loss)
        log_file.write("Evaluation of validation: average loss = %.5f\n" % avg_loss)
        log_file.flush()
        log_file.close()
    return avg_loss

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def save_model(ckpt, opts, model_name):
    save_dir = os.path.join(opts.checkpoints_dir, model_name)
    torch.save(ckpt, save_dir)

def draw_loss_curve(loss, opts):
    y_epoch = np.array(list(range(len(loss))))
    x_loss = np.array(loss)
    plt.figure()
    plt.title('loss curve')
    plt.plot(y_epoch, x_loss, color='blue', label='train loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(opts.checkpoints_dir, 'loss.png'))
    plt.close()


if __name__ == '__main__':
    # 训练网络代码
    opts = parse_opt()
    Trainer(opts)
