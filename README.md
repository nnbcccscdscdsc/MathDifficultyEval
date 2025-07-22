# 数学题难度评估项目 (Math Difficulty Evaluation)

这个项目用于评估不同参数的Llama模型在不同难度等级数学题上的表现，帮助理解模型在不同数学难度上的能力差异。

## 🎯 项目目标

1. **数据集收集**: 获取包含不同难度等级的数学题数据集
2. **模型评估**: 使用不同参数的Llama模型进行数学题解答
3. **性能分析**: 分析模型在不同难度等级上的表现差异
4. **可视化**: 生成直观的评估结果图表

## 📁 项目结构

```
MathDifficultyEval/
├── data/                    # 数据集目录
│   ├── raw/                # 原始数据集
│   └── processed/          # 处理后的数据
├── models/                 # 模型配置和权重
├── scripts/               # 核心脚本
│   ├── data_processing.py # 数据处理脚本
│   ├── model_evaluation.py # 模型评估脚本
│   ├── results_analysis.py # 结果分析脚本
│   └── utils.py           # 工具函数
├── configs/               # 配置文件
│   └── config.yaml       # 主配置文件
├── results/               # 评估结果
├── notebooks/             # Jupyter notebooks
└── requirements.txt       # 依赖包
```

## 📊 支持的数据集

### 1. MATH Dataset (hendrycks/math)
- **描述**: 包含12K个数学问题，涵盖小学到大学水平
- **难度等级**: 按年级划分 (grade_1 到 college)
- **特点**: 问题类型丰富，包含详细解答

### 2. GSM8K (Grade School Math 8K)
- **描述**: 8.5K个小学数学问题
- **难度等级**: 主要针对小学水平
- **特点**: 应用题为主，需要多步推理

### 3. MATHQA
- **描述**: 37K个数学问题，包含选择题
- **难度等级**: 涵盖基础到高级
- **特点**: 包含选项和详细解答

## 🤖 支持的模型

- **Llama-7B**: 7B参数版本
- **Llama-13B**: 13B参数版本  
- **Llama-70B**: 70B参数版本
- **量化支持**: 4bit, 8bit量化

## 📈 难度等级定义

- **小学 (Elementary)**: 
  - 基础算术运算
  - 简单应用题
  - 分数和小数
  - 基础几何

- **中学 (Middle)**: 
  - 代数方程
  - 几何证明
  - 三角函数
  - 概率统计基础

- **大学 (College)**: 
  - 微积分
  - 线性代数
  - 高等概率统计
  - 复杂数学证明

## 🚀 快速开始

### 1. 环境准备
```bash
# 创建虚拟环境
conda create -n math_eval python=3.9
conda activate math_eval

# 安装依赖
pip install -r requirements.txt
```

### 2. 快速启动（推荐）
```bash
# 使用快速启动脚本
python run_evaluation.py --model llama-7b --quantization 4bit --dataset sample --max-samples 50
```

### 3. 分步执行
```bash
# 下载和处理数据
python scripts/data_processing.py --all

# 运行评估
python scripts/model_evaluation.py --model llama-7b --quantization 4bit

# 分析结果
python scripts/results_analysis.py
```

## 📊 评估指标

- **准确率 (Accuracy)**: 正确答案的比例
- **精确匹配 (Exact Match)**: 完全匹配正确答案的比例
- **ROUGE分数**: 文本相似度评估
- **BLEU分数**: 机器翻译质量评估
- **推理步骤正确性**: 多步推理的正确性

## 📈 结果可视化

项目会自动生成以下图表：
- 不同模型在各难度等级上的准确率对比
- 模型大小与性能的关系
- 量化对模型性能的影响
- 错误类型分析

## 🔧 配置说明

主要配置文件 `configs/config.yaml` 包含：
- 模型参数设置
- 数据集配置
- 评估参数
- 输出设置

## 📝 使用示例

```python
from scripts.model_evaluation import ModelEvaluator

# 初始化评估器
evaluator = ModelEvaluator(
    model_name="llama-7b",
    quantization="4bit",
    dataset="math"
)

# 运行评估
results = evaluator.evaluate()

# 查看结果
print(f"总体准确率: {results['overall_accuracy']:.2%}")
print(f"小学难度准确率: {results['elementary_accuracy']:.2%}")
print(f"中学难度准确率: {results['middle_accuracy']:.2%}")
print(f"大学难度准确率: {results['college_accuracy']:.2%}")
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

MIT License

## 📞 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。 