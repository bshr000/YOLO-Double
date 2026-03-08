基于 PyTorch 的单模态目标检测项目，旨在完整打通目标检测框架的基本流程，并实现若干常见算法。
整体思路参考 YOLO 系列（Backebone + PAN-FPN + Decoupled Head），其中Backebone包含CspDarknet系列，Resnet系列等，Decoupled Head包含DFL回归分支与Anchor-Free风格预测。
当前实现包含训练、验证（mAP）和推理可视化三条主流程。


### 模型结构图（det_net）

![Model Architecture](pic/det_net.png)



## 1. 仓库介绍

本项目面向 YOLO 格式数据集（`class cx cy w h`，归一化坐标），支持：

- 多类别目标检测
- 可切换骨干网络（如 `resnet18/resnet34/resnet50/cspdarknet_*`）
- PAN-FPN 特征融合
- DFL 回归分支与 Anchor-Free 风格预测
- 训练阶段 EMA、AMP、梯度裁剪

核心脚本说明：

- `train.py`：模型训练
- `val.py`：验证评估（mAP@0.5 与 mAP@0.5:0.95）
- `inference.py`：离线推理与结果保存
- `dataset.py`：YOLO 数据集读取与增强
- `model.py`：模型结构定义
- `loss.py`：检测损失
- `utils.py`：通用工具函数
- `config.yaml`：训练与数据配置

## 2. 环境安装

```bash
pip install -r requirements.txt
```

## 3. 数据及数据结构

采用VEDAI数据集中的可见光模态数据(512x512，HBB版本)作为示例，原始数据集下载地址：https://downloads.greyc.fr/vedai/

或者按下面的方式组织自己的数据集：

```text
your_dataset/
  images/
    train/
      000001.png
      000002.png
    val/
      000101.png
  labels/
    train/
      000001.txt
      000002.txt
    val/
      000101.txt
```


### 3.3 配置文件

在 `config.yaml` 中修改数据路径与类别数：

- `data.train_images`
- `data.train_labels`
- `data.val_images`
- `data.val_labels`
- `data.num_classes`


## 4. 命令

### 4.1 训练

```bash
python train.py --config config.yaml
```

### 4.2 验证（mAP）

```bash
python val.py --verbose --config config.yaml --checkpoint checkpoints/best_model.pth
```

### 4.3 推理

```bash
python inference.py --save-txt --source path/to/images --config config.yaml --checkpoint checkpoints/best_model.pth --output inference_output
```

## 5. 结果示例，以YOLO-Single (CSPDarknet-S) 模型 vedai8数据集 50epoch 的权重为例

### 5.1   val

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| all | 0.640 | 0.269 | 0.423 | 0.248 |
| 0 | 0.758 | 0.724 | 0.718 | 0.414 |
| 1 | 0.723 | 0.632 | 0.649 | 0.415 |
| 2 | 0.571 | 0.512 | 0.428 | 0.271 |
| 3 | 0.345 | 0.666 | 0.393 | 0.215 |
| 4 | 0.210 | 0.200 | 0.064 | 0.023 |
| 5 | 0.545 | 0.315 | 0.260 | 0.097 |
| 6 | 0.795 | 0.235 | 0.248 | 0.146 |
| 7 | 1.000 | 0.500 | 0.620 | 0.401 |

### 5.2 inference

![Top5 inference comparison](pic/top5_compare.jpg)
  
