# 基于轻量化 YOLO 的遥感目标检测最小成本实验项目

本项目用于支撑本科论文《基于轻量化YOLO模型的星载遥感图像实时目标检测算法研究》的实验部分。目标是用最低成本跑出真实可填入论文的指标：Precision、Recall、mAP、模型大小、推理时间和 FPS。

## 1. 实验思路

推荐使用免费 GPU 平台训练，例如 Google Colab 或 Kaggle Notebook。实验采用三组模型：

| 实验 | 模型 | 输入尺寸 | 论文作用 |
|---|---:|---:|---|
| yolo11s_remote_640 | YOLO11s | 640 | 精度较高的对照组 |
| yolo11n_remote_640 | YOLO11n | 640 | 轻量化主模型 |
| yolo11n_remote_512 | YOLO11n | 512 | 更快的部署优化模型 |

论文中可以围绕这三组实验分析：YOLO11n 相比 YOLO11s 降低模型规模和推理开销，YOLO11n-512 进一步提升实时性，但可能牺牲部分小目标检测精度。

## 2. 目录结构

```text
remote_yolo_experiment/
  configs/
    remote.yaml              # 数据集配置
    experiments.yaml         # 三组实验配置
  datasets/
    remote_dataset/          # 训练数据输出目录
  reports/                   # 论文结果表输出目录
  runs/
    remote_yolo/             # YOLO 训练输出目录
  src/
    prepare_dataset.py       # 数据集整理与 VOC XML 转 YOLO
    train_experiments.py     # 批量训练三组实验
    benchmark_speed.py       # 推理速度测试
    export_models.py         # ONNX/INT8 导出
    summarize_results.py     # 结果汇总为 CSV/Markdown
```

## 3. 安装环境

```bash
cd remote_yolo_experiment
pip install -r requirements.txt
```

如果在 Colab 或 Kaggle 上运行，建议先确认 GPU：

```bash
nvidia-smi
```

## 4. 准备数据

### 方案 A：已有 YOLO 格式标注

原始数据按下面结构放置：

```text
raw/images/*.jpg
raw/labels/*.txt
```

每个标签文件格式为：

```text
class_id x_center y_center width height
```

然后执行：

```bash
python src/prepare_dataset.py --images raw/images --labels raw/labels --clean
```

### 方案 B：RSOD 等 VOC XML 标注

原始数据按下面结构放置：

```text
raw/images/*.jpg
raw/annotations/*.xml
```

然后执行：

```bash
python src/prepare_dataset.py --images raw/images --annotations raw/annotations --classes aircraft oiltank overpass playground --clean
```

默认类别为：

```text
aircraft, oiltank, overpass, playground
```

如果你的数据集类别名称不同，需要同步修改 `configs/remote.yaml` 和 `--classes` 参数。

## 5. 开始训练

完整训练：

```bash
python src/train_experiments.py --device 0
```

快速测试代码是否能跑通：

```bash
python src/train_experiments.py --device 0 --quick
```

Windows PowerShell 也可以运行：

```powershell
.\run_train.ps1 -Device 0
```

训练完成后，每组实验会生成：

```text
runs/remote_yolo/实验名/results.csv
runs/remote_yolo/实验名/results.png
runs/remote_yolo/实验名/confusion_matrix.png
runs/remote_yolo/实验名/weights/best.pt
```

## 6. 测试推理速度

```bash
python src/benchmark_speed.py --split val --samples 100 --device 0
```

输出：

```text
reports/speed_results.csv
```

该文件中的 `avg_ms_per_image` 和 `fps` 可直接填入论文“推理时间”和“FPS”。

## 7. 汇总论文结果表

```bash
python src/summarize_results.py
```

输出：

```text
reports/thesis_results.csv
reports/thesis_results.md
```

`reports/thesis_results.md` 可以直接作为论文第 5 章实验表格的草稿。

## 8. 导出部署模型

普通 ONNX：

```bash
python src/export_models.py --format onnx --device 0
```

FP16：

```bash
python src/export_models.py --format onnx --half --device 0
```

INT8：

```bash
python src/export_models.py --format onnx --int8 --device 0
```

INT8 需要校准数据，建议在完整训练数据准备好后再运行。

## 9. 论文待填项对应关系

| 论文指标 | 来源文件 | 字段 |
|---|---|---|
| Precision | reports/thesis_results.csv | precision |
| Recall | reports/thesis_results.csv | recall |
| mAP@0.5 | reports/thesis_results.csv | mAP50 |
| mAP@0.5:0.95 | reports/thesis_results.csv | mAP50_95 |
| 模型大小 | reports/thesis_results.csv | model_size_MB |
| 推理时间 | reports/thesis_results.csv | inference_ms |
| FPS | reports/thesis_results.csv | fps |
| 混淆矩阵 | runs/remote_yolo/实验名/confusion_matrix.png | 图片 |
| 训练曲线 | runs/remote_yolo/实验名/results.png | 图片 |

## 10. 建议写入论文的实验描述

本文使用公开遥感目标检测数据构建实验集，并选取 YOLO11s、YOLO11n 和 YOLO11n-512 三组模型进行对比。YOLO11s 用作精度对照模型，YOLO11n 用作轻量化主模型，YOLO11n-512 用作面向星载部署的快速推理模型。实验从检测精度、模型大小和推理速度三个方面评价模型性能，以验证轻量化 YOLO 模型在星载遥感图像实时目标检测任务中的可行性。
