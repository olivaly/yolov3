## Yolov3: Yolov3: An Incremental Improvement 论文复现

# 代码为yolov3的论文复现，目前只支持单卡训练

# 环境： torch==2.2.0
环境具体配置参见requirements.txt

# 训练准备：
1.配置环境
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

2.数据准备
代码采用工业缺陷检测数据neu数据集进行训练，下载地址如下：
链接：https://pan.baidu.com/s/1y83e6UfYCXvmrRYPPhL4yw?pwd=w3t7 
提取码：w3t7

提取码：
数据集已经划分了训练集、验证集，无需再次划分。

3.网路训练
修改默认参数中的数据集路径"dataset_dir"，保存路径"checkpoints_dir"，运行train.py开始训练， 

4.测试结果
修改测试数据路径"dataset_dir"，检测结果保存路径"save_path"，模型参数路径"checkpoints_dir"，运行predict.py进行预测。

Reference:
https://arxiv.org/abs/1804.02767
https://github.com/ultralytics/yolov3
https://github.com/bubbliiiing/yolo3-pytorch
