# OpenAI评分功能使用指南

## 概述

本项目已集成OpenAI API评分功能，可以对不同参数的Llama模型在数学题上的表现进行专业评分，并生成类似您手绘图的性能曲线。

## 功能特点

✅ **专业评分**: 使用GPT-3.5-turbo对答案进行多维度评分
✅ **多维度评估**: 答案正确性、解题思路、步骤完整性、表达清晰度
✅ **性能曲线**: 自动生成模型参数与性能的关系曲线
✅ **批量处理**: 支持批量评分，自动处理API限制
✅ **详细报告**: 生成完整的评估报告和可视化图表

## 安装依赖

```bash
# 安装OpenAI API
pip install openai==0.28.0

# 设置API密钥
export OPENAI_API_KEY="your_openai_api_key_here"
```

## 配置说明

### 1. 环境变量设置

```bash
# 设置OpenAI API密钥
export OPENAI_API_KEY="sk-your-api-key-here"
```

### 2. 配置文件修改

`configs/config.yaml` 中的OpenAI配置：

```yaml
# OpenAI评分配置
openai_scoring:
  enabled: true
  model: "gpt-3.5-turbo"
  temperature: 0.0
  max_tokens: 100
  prompt_template: |
    你是一个专业的数学教育评估专家。请对以下数学问题的回答进行评分。
    
    问题: {problem}
    标准答案: {reference_answer}
    学生回答: {student_answer}
    
    请从以下几个方面进行评分（总分100分）：
    1. 答案正确性（40分）：答案是否正确
    2. 解题思路（30分）：解题思路是否清晰合理
    3. 步骤完整性（20分）：解题步骤是否完整
    4. 表达清晰度（10分）：表达是否清晰易懂
    
    请只返回一个0-100之间的数字分数，不要其他内容。
```

## 使用方法

### 1. 单模型评估（带OpenAI评分）

```bash
# 评估单个模型
python scripts/model_evaluation.py --model llama-7b --dataset sample --max-samples 10
```

### 2. 多模型比较（生成性能曲线）

```bash
# 比较多个模型
python scripts/multi_model_comparison.py --models llama-7b llama-13b --dataset sample --max-samples 20
```

### 3. 结果分析

```bash
# 分析单个模型结果
python scripts/results_analysis.py --results-file results/llama-7b_sample_*.csv --model-name llama-7b
```

## 输出结果

### 1. 评分结果

每个评估结果包含：
- `openai_score`: 0-100的评分
- `openai_score_text`: 评分详情
- 其他传统指标（准确率、ROUGE等）

### 2. 可视化图表

自动生成以下图表：
- **性能对比图**: 不同模型在各指标上的对比
- **参数曲线图**: 模型参数与性能的关系曲线（类似手绘图）
- **难度分析图**: 各难度等级的性能分布
- **交互式图表**: 可交互的HTML格式图表

### 3. 报告文件

- **Markdown报告**: 详细的评估报告
- **CSV数据**: 原始评估数据
- **JSON摘要**: 统计摘要

## 性能曲线说明

生成的性能曲线将展示：

1. **X轴**: 模型参数数量（7B, 13B, 70B）
2. **Y轴**: OpenAI评分（0-100）
3. **曲线**: 显示参数增加与性能提升的关系
4. **多条曲线**: 不同难度等级的性能曲线

预期结果类似您的手绘图：
- 曲线从原点开始上升
- 随着参数增加，性能提升但逐渐平缓
- 不同难度等级有不同的性能表现

## 评分标准

OpenAI评分器从四个维度进行评分：

1. **答案正确性（40分）**: 最终答案是否正确
2. **解题思路（30分）**: 解题思路是否清晰合理
3. **步骤完整性（20分）**: 解题步骤是否完整
4. **表达清晰度（10分）**: 表达是否清晰易懂

## 成本控制

### 1. API调用优化

- 使用批量处理减少API调用次数
- 设置合理的延迟避免速率限制
- 缓存评分结果避免重复调用

### 2. 成本估算

假设每个问题评分消耗100 tokens：
- 100个问题 ≈ 10,000 tokens
- 使用GPT-3.5-turbo ≈ $0.002

### 3. 替代方案

如果不想使用OpenAI API，可以：
- 设置 `enabled: false` 禁用OpenAI评分
- 仅使用传统指标（准确率、ROUGE等）
- 使用本地评分模型

## 故障排除

### 1. API密钥问题

```bash
# 检查API密钥
echo $OPENAI_API_KEY

# 测试API连接
python scripts/openai_scorer.py --test
```

### 2. 评分失败

- 检查网络连接
- 确认API配额充足
- 查看日志文件了解详细错误

### 3. 内存不足

- 减少批处理大小
- 使用量化模型
- 增加系统内存

## 示例输出

### 评分结果示例

```json
{
  "id": "test_1",
  "problem": "What is 2 + 3?",
  "generated_answer": "The answer is 5.",
  "openai_score": 85.0,
  "openai_score_text": "85",
  "accuracy": 0.8,
  "difficulty": "elementary"
}
```

### 性能曲线数据

```csv
model,parameters,difficulty,avg_openai_score
llama-7b,7,elementary,75.2
llama-7b,7,middle,68.5
llama-7b,7,college,62.1
llama-13b,13,elementary,82.3
llama-13b,13,middle,76.8
llama-13b,13,college,71.4
```

## 下一步

1. **设置API密钥**: `export OPENAI_API_KEY="your-key"`
2. **运行测试**: `python scripts/openai_scorer.py --test`
3. **评估模型**: `python scripts/multi_model_comparison.py --models llama-7b llama-13b`
4. **查看结果**: 检查 `results/` 目录下的图表和报告

这样您就能获得类似手绘图的性能曲线，展示不同参数模型在不同难度数学题上的表现！ 