# HiCroPL-Independent-RS

面向遥感图像分类的独立双分支深层提示学习基线。从 `applenet_baseline` 分离，仅运行 **HiCroPL_NoFusion**。项目名用于标识研究工程，不代表新提出的方法或完整 HiCroPL。

模型和训练流程保持原项目行为：移除双向提示映射及代理池化，保留视觉/文本独立深层提示、冻结特征残差和双分支余弦蒸馏。本次分离不调整训练超参数，也不宣称提高精度。原项目保留，不额外创建完整备份。

## 保留与移除

| 内容 | 处理 |
|---|---|
| NoFusion 提示学习器、编码器、损失、优化与加载逻辑 | 保留；仅将 CLIP 加载函数从 CoOp 文件迁入公共文件 |
| 8 个遥感数据集加载器、划分 JSON | 原样保留 |
| 固定模板配置、8 份描述版配置、类别描述 JSON | 原样保留 |
| Base-to-New、跨数据集、v2 跨域入口 | 保留；默认方法改为 NoFusion |
| CoOp、PromptSRC 实现、配置与专属测试 | 移除 |
| Dassl 其他方法与无关数据集的注册、专属配置 | 移除；保留通用训练设施 |
| 来源与许可证 | 保留；历史生成和一次性验证材料已清理 |

内部标识继续使用 `HiCroPLNoFusion`、`TRAINER.HICROPL_NF` 和 `--method hicropl_nofusion`，保证旧命令及 NoFusion 提示 checkpoint 的兼容性。现在可以省略 `--method`，不会运行其他方法。

## 环境

已验证 Python 3.11、PyTorch 2.3.1、torchvision 0.18.1。先安装与服务器 CUDA 匹配的 PyTorch/torchvision，再安装本项目依赖：

```bash
python -m pip install -r requirements.txt
```

沿用服务器原有环境更利于比较分离前后的成绩。不要在这次对照中同时升级 PyTorch、改变批量或切换计算精度。

## 服务器首次复跑

上传整个项目文件夹。在服务器中进入项目根目录，将以下路径替换为服务器上的真实位置：

```bash
cd /你的项目路径/HiCroPL-Independent-RS
DATA_ROOT=/你的数据集父目录
CLIP_WEIGHTS=/你的权重路径/ViT-B-16.pt

# 只打印命令，先检查数据、配置和输出位置。
python -B scripts/run.py base2new train rsicd 1 --root "$DATA_ROOT" --weights "$CLIP_WEIGHTS" --dry-run

# 固定三模板版本：训练基础类别，再分别测试基础/新类别。
python -u -B scripts/run.py base2new train rsicd 1 --root "$DATA_ROOT" --weights "$CLIP_WEIGHTS"
python -u -B scripts/run.py base2new test rsicd 1 --root "$DATA_ROOT" --weights "$CLIP_WEIGHTS" --subset base
python -u -B scripts/run.py base2new test rsicd 1 --root "$DATA_ROOT" --weights "$CLIP_WEIGHTS" --subset new
```

`DATA_ROOT` 下应有 `RSICD/images/`，而不是直接指向 `images`。也支持 `patternnet`、`resisc45`、`mlrsnet` 及各自的 `v2` 数据集。

默认输出相对于本项目根目录：

```text
outputs/base2new/train_base/rsicd/shots_16/HiCroPLNoFusion/vit_b16_c16/seed1/
outputs/base2new/test_base/rsicd/shots_16/HiCroPLNoFusion/vit_b16_c16/seed1/
outputs/base2new/test_new/rsicd/shots_16/HiCroPLNoFusion/vit_b16_c16/seed1/
```

模型从基础类别验证集选择，测试默认加载 `prompt_learner/model-best.pth.tar`。显式 `--load-epoch 50` 可加载第 50 轮；请与旧实验选模规则保持一致。原项目的日志、训练结果、预训练权重和数据图片不包含在新项目中。

### 如果之前使用每类 20 条描述

必须继续选择对应描述版配置，否则教师文本变化会影响精度。以 RSICD 为例，训练及两次测试命令均加：

```text
--config vit_b16_c16_rsicd_desc20
```

其他数据集使用 `vit_b16_c16_数据集名_desc20`。描述文件由 Codex 助手根据类别语义撰写，每类 20 条，不是 HiCroPL 作者的原始 GPT 输出；未使用测试图片或逐图标注编写。

## 一个脚本运行三种正式实验

`scripts/run_all.py` 使用 desc20 和 seed 1、2、3，按现有配置顺序运行全部任务：

| 实验 | 训练与测试安排 | 训练次数 | 独立评估次数 |
|---|---|---:|---:|
| Base-to-New | PatternNet、RSICD、RESISC45、MLRSNet 分别训练 Base，再测试 Base/New | 12 | 24 |
| Cross-Datasets | PatternNet 全类别 16-shot 训练，测试 RSICD、RESISC45、MLRSNet | 3 | 9 |
| SSMT | PatternNetv2 的共享 16 类训练，测试 RSICDv2、RESISC45v2、MLRSNetv2 | 3 | 9 |
| 合计 | 顺序运行，沿用原 YAML 的优化器、轮数及精度 | 18 | 42 |

SSMT 使用对应的 v2 数据集；原版数据集类别数不同，不能直接替代。三种实验仍采用 16-shot、50 轮、fp32、batch=4 和 best_val；本次只新增调度脚本，没有修改方法及训练设置。

在服务器项目根目录，替换下面两个实际路径后运行：

```bash
python -u -B scripts/run_all.py \
  --root /服务器数据集父目录 \
  --output-root outputs/all_desc20_seed123
```

总脚本的 `--weights` 可省略：优先使用项目 `.cache/clip/ViT-B-16.pt`，其次用户主目录 `.cache/clip/ViT-B-16.pt`；使用官方校验和验证缓存，没有可用缓存时联网下载到项目缓存。已有自定义权重时仍可显式传入 `--weights /实际路径/ViT-B-16.pt`。首次下载需要服务器能够连接下载地址，后续复用缓存。

如需先检查任务列表，追加 `--dry-run`；该选项不训练，也不创建输出文件。正式任务建议在 tmux 中运行。通过环境变量 `CUDA_VISIBLE_DEVICES=0` 可指定显卡；脚本依次执行，不会同时开启多个训练。

数据位置支持统一父目录，以及旧布局中的 `APPLeNet_PN/PatternNet`、`APPLeNet_RSICD/RSICD` 和 `NWPU Data Set/RESISC45`。特殊布局可显式追加加载器的父目录，例如：

```text
--dataset-root rsicd=/data/APPLeNet_RSICD
```

需要全部 8 个数据集目录的 `images/`。正式执行前会检查目录、划分、描述覆盖及 v2 的共享 16 类。它不会逐张检查所有图片内容。

输出根目录应为新目录，防止误用旧模型。完成后查看：

- `results.csv`：各实验逐种子结果，Base-to-New 含 Base、New、H。
- `summary.csv`：3 个种子的均值和样本标准差；H 先逐种子计算再平均。
- `run_plan.json`、`status.json`：本轮任务配置、文件校验和及执行进度。
- 各任务目录：原生配置、日志和 checkpoint；Cross-Datasets 仍保存在 `crossdata/`，SSMT 保存在 `domaingen/`，以兼容现有启动器。

出错或手动中断后，修复原因并用相同命令追加 `--resume`：

```bash
python -u -B scripts/run_all.py \
  --root /服务器数据集父目录 \
  --output-root outputs/all_desc20_seed123 \
  --resume
```

已完成任务直接跳过。中断的训练会将其旧目录重命名为 `seedN.interrupted_时间戳` 后，从相同 seed 重新开始该次训练；不沿用未完整保存的随机数和选模状态。勿在前一个进程仍运行时启动 `--resume`。恢复时若代码、权重、划分或描述发生变化，脚本会拒绝混用结果，需使用新的输出目录。

## 默认训练设置

| 设置 | 数值 |
|---|---|
| 骨干 / 输入 | CLIP ViT-B/16 / 224×224 |
| 提示深度 / 每层长度 | 12 / 16 |
| 可训练参数 | 245,760；仅视觉与文本提示 |
| 优化器 / 学习率 / Adam epsilon | Adam / 0.0025 / 0.001 |
| 轮数 / 调度 | 50 / cosine，1 轮 constant warmup |
| 蒸馏系数 | 12 |
| 训练 / 测试 batch，workers | 4 / 32，0 |
| 精度 / shots / seed | fp32 / 16 / 命令指定 |
| 教师文本 | 默认 3 条固定遥感模板；描述版显式指定 JSON |
| 选模 | best_val |

实际设置以每次输出目录的 `config.yaml` 为准。与旧结果比较时，除 seed 和配置外，还要保证同一数据划分、few-shot 缓存、权重文件、描述文件、预处理及计算环境。数据加载器优先使用数据集目录中的划分，再使用本项目附带划分；少样本缓存仍位于数据集的 `shots/` 目录。

## 其他协议与旧模型

```bash
# 跨数据集：PatternNet 全类别训练，迁移到 RSICD。
python -u -B scripts/run.py crossdata train patternnet 1 --root "$DATA_ROOT" --weights "$CLIP_WEIGHTS"
python -u -B scripts/run.py crossdata test rsicd 1 --source patternnet --root "$DATA_ROOT" --weights "$CLIP_WEIGHTS"

# v2 对齐类别的跨域任务。
python -u -B scripts/run.py domaingen train patternnetv2 1 --root "$DATA_ROOT" --weights "$CLIP_WEIGHTS"
python -u -B scripts/run.py domaingen test rsicdv2 1 --source patternnetv2 --root "$DATA_ROOT" --weights "$CLIP_WEIGHTS"
```

迁移时保留源训练的 `--config` 名以定位 checkpoint。若源训练用描述版，测试还需在命令末尾覆盖目标描述路径：`-- TRAINER.HICROPL_NF.TEXT_PROMPTS gpt_files/rsicd_prompt.json`。

直接测试旧项目的 NoFusion 提示模型（以下是固定模板示例；描述版改用原训练对应配置）：

```bash
python -B train.py --root "$DATA_ROOT" --seed 1 --dataset-config-file yaml/datasets/rsicd.yaml --config-file yaml/trainers/HiCroPLNoFusion/vit_b16_c16.yaml --eval-only --model-dir /旧训练输出目录 --output-dir outputs/old_checkpoint_new DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES new MODEL.CLIP_WEIGHTS "$CLIP_WEIGHTS"
```

`--model-dir` 指向包含 `prompt_learner/` 的目录。内部 checkpoint 参数键未改；完整原版 HiCroPL 的整模型 checkpoint 不属于此兼容范围。续训使用 `train.py --resume /旧训练输出目录` 并带上原来的数据/配置参数。

## 服务器验收

分离时已在本机验证原/新项目的模型输出、梯度及一次参数更新逐值一致，包括真实 ViT-B/16。一次性验证材料已从训练项目清理。服务器仍需使用相同设置正式复跑，评估准确率及随机波动。

## 代码结构

```text
models/hicropl_nofusion.py       独立提示、教师特征、损失与训练器
models/clip_loader.py            提取出的公共 CLIP 权重加载函数
clip/hicropl_model.py            独立深层提示编码器
clip/                           CLIP 基础网络与 tokenizer
dassl/                          通用训练、数据、评估与优化设施
files/ + json/                  遥感加载器与原数据划分
yaml/trainers/HiCroPLNoFusion/   固定模板及 8 份描述版配置
gpt_files/                      8 份训练可用的遥感类别描述 JSON
scripts/run.py                  单次实验启动
scripts/run_all.py              三种协议、四个数据集、三种子的总调度
scripts/*.sh                    单次实验的便捷入口
```

## 来源

方法基于 [HiCroPL](https://github.com/zzeoZheng/HiCroPL)，遥感实验设施来源于 [APPLeNet](https://github.com/mainaksingha01/APPLeNet)，基础库包括 [CLIP](https://github.com/openai/CLIP)、[Dassl](https://github.com/KaiyangZhou/Dassl.pytorch)；保留相关 CoOp 派生代码归属。许可证见 `LICENSE`、`licenses/`。本项目删除双向交互且更改教师文本来源，应作为 HiCroPL 衍生研究基线报告。

描述文件中的类别键与数据加载器一致，下划线按原规则转换为空格；部分同义类别和 v2 子集共用描述。教师特征按每句归一化、同类求平均、再次归一化的顺序计算。清理未改动描述 JSON、配置、数据划分或训练代码。
