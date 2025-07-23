# 数学难度评估项目

## 项目概述

本项目用于评估不同大语言模型在数学问题上的表现，支持多GPU推理和OpenAI评分。

## 当前状态

### ✅ 已完成功能
- 多模型评估框架
- 模型配置管理系统
- 数据集处理（DeepMath数据集）
- OpenAI评分集成
- 多GPU支持

### ⚠️ 当前问题
1. **模型推理输出质量差**：测试的Mistral-7B模型输出异常内容
2. **需要更好的模型选择**：可能需要使用专门训练过的数学模型

### 🔧 技术栈
- Python 3.10
- PyTorch + Transformers
- BitsAndBytes量化
- OpenAI API
- Pandas + NumPy

## 快速测试

```bash
# 激活环境
conda activate math_eval_env

# 快速推理测试
python quick_test.py
```

## 项目结构

```
├── configs/           # 模型配置文件
├── data/             # 数据集
├── results/          # 评估结果
├── scripts/          # 核心脚本
│   ├── model_evaluation.py      # 模型评估核心
│   ├── evaluate_single_model.py # 单模型评估
│   ├── evaluate_all_models.py   # 多模型评估
│   └── quick_evaluation.py      # 快速评估
└── quick_test.py     # 简单推理测试
```

## 使用方法

### 1. 单模型评估
```bash
python scripts/evaluate_single_model.py --model mistral-7b
```

### 2. 多模型评估
```bash
python scripts/evaluate_all_models.py --parallel
```

### 3. 快速评估（带OpenAI评分）
```bash
python scripts/quick_evaluation.py --model mistral-7b --openai
```

## 配置说明

每个模型都有独立的YAML配置文件，包含：
- GPU配置
- 量化设置
- 生成参数
- 提示模板

## 问题排查

### 常见问题
1. **CUDA错误**：检查GPU内存和模型大小
2. **推理输出异常**：调整生成参数或更换模型
3. **OpenAI评分失败**：检查API密钥和网络连接

### 调试建议
- 使用 `quick_test.py` 进行基础推理测试
- 检查模型配置文件中的参数
- 查看日志输出中的错误信息

## 下一步计划

1. 测试更多数学专用模型（如MathGPT、WizardMath等）
2. 优化生成参数和提示模板
3. 增加更多评估指标
4. 改进结果分析和可视化

## 环境要求

- CUDA 11.8+
- Python 3.10+
- 至少16GB GPU内存（用于大模型）
- OpenAI API密钥（用于评分） 