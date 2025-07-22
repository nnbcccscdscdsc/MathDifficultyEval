# 支持的模型列表

本项目现在支持以下公开的大语言模型，可以直接使用而无需申请特殊权限。

## 模型概览

| 模型名称 | 参数大小 | 上下文长度 | 特点 | 推荐用途 |
|---------|---------|-----------|------|---------|
| Mistral-7B-v0.2 | 7B | 4K | 高质量开源模型 | 通用任务 |
| LongAlpaca-7B-16k | 7B | 16K | 长文本处理能力强 | 长文档理解 |
| LongAlpaca-13B-16k | 13B | 16K | 中等规模，性能平衡 | 复杂推理 |
| LongAlpaca-70B-16k | 70B | 16K | 大规模，性能最佳 | 高精度任务 |
| OASST-30B | 30B | 4K | RLHF优化，对话友好 | 对话和问答 |

## 详细模型信息

### 1. Mistral-7B-v0.2
- **模型ID**: `mistral-community/Mistral-7B-v0.2`
- **参数大小**: 7B
- **上下文长度**: 4,096 tokens
- **特点**: 
  - 高质量的开源模型
  - 在多个基准测试中表现优秀
  - 适合通用任务和数学推理
- **推荐场景**: 数学题解答、文本生成、代码生成

### 2. LongAlpaca-7B-16k
- **模型ID**: `Yukang/LongAlpaca-7B-16k`
- **参数大小**: 7B
- **上下文长度**: 16,384 tokens
- **特点**:
  - 基于Alpaca训练
  - 支持超长文本处理
  - 适合需要长上下文的数学问题
- **推荐场景**: 复杂数学问题、多步骤推理

### 3. LongAlpaca-13B-16k
- **模型ID**: `Yukang/LongAlpaca-13B-16k`
- **参数大小**: 13B
- **上下文长度**: 16,384 tokens
- **特点**:
  - 中等规模模型
  - 性能与效率的平衡
  - 长文本处理能力强
- **推荐场景**: 中等复杂度数学问题、需要详细推理的任务

### 4. LongAlpaca-70B-16k
- **模型ID**: `Yukang/LongAlpaca-70B-16k`
- **参数大小**: 70B
- **上下文长度**: 16,384 tokens
- **特点**:
  - 大规模模型，性能最佳
  - 超长上下文支持
  - 适合复杂推理任务
- **推荐场景**: 高难度数学问题、复杂推理、需要最高精度的任务

### 5. OASST-30B
- **模型ID**: `Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf`
- **参数大小**: 30B
- **上下文长度**: 4,096 tokens
- **特点**:
  - 经过RLHF优化
  - 对话友好
  - 适合问答和解释
- **推荐场景**: 数学概念解释、教学问答、详细解答

## 使用建议

### 按任务复杂度选择

1. **简单数学题** (小学水平)
   - 推荐: `mistral-7b` 或 `longalpaca-7b`
   - 原因: 7B参数足够处理基础数学问题

2. **中等复杂度** (中学水平)
   - 推荐: `longalpaca-13b` 或 `oasst-30b`
   - 原因: 需要更强的推理能力

3. **高复杂度** (大学水平)
   - 推荐: `longalpaca-70b`
   - 原因: 大规模模型处理复杂数学问题效果最佳

### 按资源限制选择

1. **GPU内存有限** (8GB以下)
   - 推荐: `mistral-7b` 或 `longalpaca-7b`
   - 使用4bit量化

2. **GPU内存充足** (16GB以上)
   - 推荐: `longalpaca-13b` 或 `oasst-30b`
   - 使用4bit或8bit量化

3. **GPU内存充足** (24GB以上)
   - 推荐: `longalpaca-70b`
   - 使用4bit量化

### 按上下文需求选择

1. **短问题** (单题解答)
   - 推荐: `mistral-7b` 或 `oasst-30b`
   - 4K上下文足够

2. **长问题** (多步骤、多题组合)
   - 推荐: `longalpaca-7b/13b/70b`
   - 16K上下文支持

## 使用示例

### 单模型评估
```bash
# 评估Mistral-7B
python scripts/model_evaluation.py --model mistral-7b --dataset sample --max-samples 10

# 评估LongAlpaca-70B
python scripts/model_evaluation.py --model longalpaca-70b --dataset sample --max-samples 10
```

### 多模型比较
```bash
# 比较不同规模的LongAlpaca模型
python scripts/multi_model_comparison.py --models longalpaca-7b longalpaca-13b longalpaca-70b --dataset sample --max-samples 20

# 比较不同架构的模型
python scripts/multi_model_comparison.py --models mistral-7b longalpaca-7b oasst-30b --dataset sample --max-samples 20
```

## 性能预期

基于模型参数大小，预期性能排序（从高到低）：
1. `longalpaca-70b` (70B参数)
2. `oasst-30b` (30B参数)
3. `longalpaca-13b` (13B参数)
4. `longalpaca-7b` (7B参数)
5. `mistral-7b` (7B参数，但架构优化)

## 注意事项

1. **内存需求**: 模型越大，GPU内存需求越高
2. **推理速度**: 模型越大，推理速度越慢
3. **量化影响**: 使用量化会轻微影响性能，但大幅减少内存需求
4. **首次下载**: 首次使用需要下载模型权重，可能需要较长时间

## 故障排除

如果遇到模型加载问题：
1. 检查网络连接
2. 确认有足够的磁盘空间存储模型
3. 检查GPU内存是否足够
4. 尝试使用量化减少内存需求 