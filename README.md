# BERT 中文标点恢复

本项目使用 `google-bert/bert-base-chinese` 微调逐字标点分类器，并比较固定长度批处理窗口与串行长文窗口。训练、验证和测试按文章划分；测试指标只用于本语料内评估。

本次实验的代码及相关说明可在该仓库中获取：[LLMandAppliedPractices-BERT-Punctuation](https://github.com/stq1234/LLMandAppliedPractices-BERT-Punctuation)。

## 环境

实验实测环境为 Python 3.12、PyTorch 2.12.0+cu126、Transformers 4.51.3、NumPy 1.26.4、Matplotlib 3.9.2、safetensors 0.5.3、huggingface-hub 0.31.2 和 tokenizers 0.21.1；GPU 为 NVIDIA GeForce RTX 3060 Laptop GPU（6 GiB）。需使用 NVIDIA CUDA 12.6 对应的 PyTorch wheel。可先按 PyTorch 官方安装说明选定 CUDA 12.6 wheel，再安装其余锁定依赖：

```powershell
python -m pip install torch==2.12.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

首次训练和评估需要能够下载 Hugging Face 模型。模型名称固定为 `google-bert/bert-base-chinese`，revision 固定为 `8f23c25b06e129b6c986331a13d8d025a92cf0ea`。微调后的约 407 MB 权重位于 `checkpoints/best_model/`，按 `.gitignore` 排除；本次权重文件 SHA-256 为 `85693e35e4033b4efa3b70edcaa9ed8aa3042c40e7d1b1cf0b0f4aab0d20091c`，完整文件清单见 `results/checkpoint_manifest.json`。

## 数据与重现

仓库包含从 1998 年 1 月《人民日报》语料抽取的 GB18030 原始文本及固定的 train/validation/test JSONL。原始文本 SHA-256 为 `37b85555df2f72642ff9815ac01f51d4f50024d80ca0e2e8a6be6f93b1038639`。重建清洗数据并核验分割：

```powershell
python src/prepare_data.py
```

完整三轮微调（CUDA 可用时运行）：

```powershell
python src/train.py --batch-size 8 --gradient-accumulation 1 --epochs 3
```

训练命令从固定 revision 加载预训练权重；输出训练曲线数据、最佳验证模型、测试预测和测试指标。测试集不参与 checkpoint、超参数或窗口参数选择。

长文窗口验证集网格选择、测试集三次交错计时和制图：

```powershell
python src/benchmark_long.py
python src/plot_results.py
```

用本地最佳 checkpoint 对 UTF-8 文件恢复标点（默认自动选择 CUDA 或 CPU；窗口固定为 256 字符、步长 128；batch size 可设为 1–16）：

```powershell
python src/punctuate.py --input data/analects_xueer_first_three_punctuation_free.txt --output results/analects_qualitative_output.txt --batch-size 4 --device cpu
```

应用入口保留输入文件中的每个原字符及其顺序，只在其后添加模型预测标点；输入应为去除目标标点后的字符骨架。窗口规划和重叠概率融合复用 `src/benchmark_long.py` 中的实现，并在覆盖或字符数不一致时抛出错误。

脚本默认比较 batch size 1、4、8、16。长文测试子集限于测试文章中长度不低于 800 字的篇目；它只服务于工程速度及长文本策略诊断，完整测试指标另见 `results/test_metrics.json`。

## 文件

- `src/prepare_data.py`：原始语料清洗、对齐、去重、文章级分割与数据审计。
- `src/train.py`：预训练编码器微调、验证集模型选择、完整测试评估和逐字预测输出。
- `src/benchmark_long.py`：长文窗口参数验证选择、串行基线与固定窗口批处理计时。
- `src/punctuate.py`：从 UTF-8 文件加载本地最佳 checkpoint，以固定 256/128 窗口与可配置 batch size 输出恢复文本。
- `src/plot_results.py`：由已记录指标与混淆矩阵生成训练、逐类 F1、混淆矩阵和长文吞吐图。
- `data/`：原始 1 月文本和冻结后的三个数据划分。
- `results/`：数据、模型、指标、错误样例及长文基准的审计材料。
- `figures/`：报告使用的训练、逐类 F1、逐类混淆和长文吞吐图。
- `report.md`：实验过程、量化结果、误差分析与适用范围。
- `results/analects_qualitative_inference.json`：公开领域古文来源、原始输入、模型输出与定性推理配置；不包含 gold F1。

模型在 1998 年报刊语料的文章级留出集上评估；本实验没有构造古文标注测试集，因此不能据此推断古文标点恢复效果，也未进行跨时期泛化评估。
