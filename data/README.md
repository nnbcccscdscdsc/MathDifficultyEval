# Data 文件夹说明

这个文件夹用于存放所有生成的数据文件和结果。

## 📁 文件夹结构

```
data/
├── processed/           # 处理后的数据集
│   └── deepmath_evaluation_dataset.csv
├── results/            # 最终评估结果
│   └── final_evaluation_*.json
├── intermediate/       # 中间结果（每20个样本保存一次）
│   └── intermediate_results_*.json
├── plots/             # 生成的图表
│   └── difficulty_score_plot_*.png
└── README.md          # 本文件
```

## 📊 文件说明

### results/ 文件夹
- **final_evaluation_*.json**: 完整的评估结果
- 包含所有样本的详细评估信息
- 包括统计摘要和难度分析

### intermediate/ 文件夹  
- **intermediate_results_*.json**: 中间结果文件
- 每处理20个样本自动保存一次
- 用于断点续传和进度跟踪

### plots/ 文件夹
- **difficulty_score_plot_*.png**: 难度-评分曲线图
- 显示模型在不同难度题目上的表现
- 包含样本分布统计

## 🔧 自动管理

- 所有文件会自动保存到对应文件夹
- 文件夹不存在时会自动创建
- 文件名包含时间戳，避免覆盖

## 📈 数据格式

### 评估结果格式
```json
{
  "id": "样本ID",
  "problem": "数学题目",
  "correct_answer": "标准答案", 
  "model_response": "模型回答",
  "difficulty": "难度等级",
  "evaluation": {
    "overall_score": 8.5,
    "answer_correctness": 9.0,
    "reasoning_logic": 8.0,
    "step_completeness": 8.5,
    "mathematical_accuracy": 9.0,
    "expression_clarity": 8.0
  }
}
```

### 统计摘要格式
```json
{
  "summary": {
    "total_evaluated": 200,
    "success_rate": 0.95,
    "average_overall_score": 8.2
  },
  "statistics": {
    "average_scores": {
      "min": 5.0,
      "max": 10.0,
      "mean": 8.2,
      "std": 1.1
    }
  }
}
``` 