# Math Difficulty Evaluation System - Organized

## 📁 目录结构 (Directory Structure)

```
MathDifficultyEval/
├── main.py                          # 主入口脚本 (Main entry script)
├── README.md                        # 原始说明文档
├── README_organized.md              # 本文件 (This file)
├── requirements.txt                 # Python依赖包
├── .gitignore                       # Git忽略文件
│
├── scripts/                         # 脚本目录 (Scripts directory)
│   ├── plotting/                    # 绘图脚本 (Plotting scripts)
│   │   ├── plot_all_datasets_weighted.py      # 权重方案对比图
│   │   ├── unified_model_analysis.py          # 统一模型分析
│   │   ├── plot_math500_*.py                  # MATH-500相关绘图
│   │   ├── plot_hendrycks_math_*.py           # Hendrycks Math相关绘图
│   │   └── plot_deepmath_103k_*.py            # DeepMath-103K相关绘图
│   │
│   ├── evaluation/                  # 评估脚本 (Evaluation scripts)
│   │   ├── unified_math_evaluation.py         # DeepMath-103K评估
│   │   ├── unified_math_evaluation_math500.py # MATH-500评估
│   │   ├── unified_math_evaluation_hendrycks.py # Hendrycks Math评估
│   │   ├── math_evaluation_framework.py       # 评估框架
│   │   └── math500_evaluation_framework.py    # MATH-500评估框架
│   │
│   └── utils/                       # 工具脚本 (Utility scripts)
│       └── create_balanced_dataset.py         # 创建平衡数据集
│
├── data/                            # 数据目录 (Data directory)
│   ├── DeepMath-103K_result/        # DeepMath-103K结果
│   ├── hendrycks_math_results/      # Hendrycks Math结果
│   ├── math500_results/             # MATH-500结果
│   └── ...
│
├── plot_data/                       # 生成的图表 (Generated plots)
├── results/                         # 评估结果 (Evaluation results)
├── configs/                         # 配置文件 (Configuration files)
└── Download/                        # 下载脚本 (Download scripts)
```

## 🚀 快速开始 (Quick Start)

### 1. 运行主程序 (Run Main Program)
```bash
python main.py
```

### 2. 直接运行特定功能 (Run Specific Function)
```bash
# 生成权重对比图
python scripts/plotting/plot_all_datasets_weighted.py

# 生成统一模型分析
python scripts/plotting/unified_model_analysis.py

# 运行MATH-500评估
python scripts/evaluation/unified_math_evaluation_math500.py
```

## 📊 主要功能 (Main Features)

### 1. 权重方案对比 (Weighting Scheme Comparison)
- **方法一**: 仅考虑答案正确性 (Answer Correctness Only)
- **方法二**: 答案正确性+推理逻辑性+步骤完整性 (Answer + Reasoning + Steps)
- **方法三**: 四项评分综合加权 (Four Criteria Weighted)

### 2. 数据集支持 (Supported Datasets)
- **DeepMath-103K**: 103K数学问题数据集
- **Hendrycks Math**: Hendrycks数学数据集
- **MATH-500**: 500题数学数据集

### 3. 模型支持 (Supported Models)
- DeepSeek-R1-Distill-Qwen-1.5B
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Qwen-14B
- DeepSeek-R1-Distill-Qwen-32B
- DeepSeek-R1-Distill-Llama-70B

## 🎯 使用建议 (Usage Recommendations)

1. **首次使用**: 运行 `python main.py` 选择功能
2. **批量分析**: 直接运行对应的脚本文件
3. **自定义分析**: 修改 `scripts/` 目录下的脚本

## 📝 注意事项 (Notes)

- 所有图表保存在 `plot_data/` 目录
- 评估结果保存在 `results/` 目录
- 确保数据目录结构正确
- 建议使用虚拟环境运行

## 🔧 故障排除 (Troubleshooting)

1. **路径问题**: 确保在项目根目录运行脚本
2. **依赖问题**: 运行 `pip install -r requirements.txt`
3. **数据问题**: 检查 `data/` 目录下的数据文件是否存在

## 📞 支持 (Support)

如有问题，请检查：
1. Python版本 (建议3.8+)
2. 依赖包安装情况
3. 数据文件完整性
4. 文件路径正确性 