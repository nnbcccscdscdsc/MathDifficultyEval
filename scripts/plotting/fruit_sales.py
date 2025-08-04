import matplotlib.pyplot as plt

# 数据准备
categories = ['苹果', '香蕉', '橙子', '梨', '葡萄']
values = [35, 28, 45, 20, 38]

# 创建柱形图
plt.figure(figsize=(8, 6))  # 设置图形大小
bars = plt.bar(categories, values, color=['red', 'yellow', 'orange', 'green', 'purple'])

# 添加标题和标签
plt.title('水果销量统计', fontsize=16)
plt.xlabel('水果种类', fontsize=12)
plt.ylabel('销量(千克)', fontsize=12)

# 在每个柱子上方显示数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

# 设置y轴范围
plt.ylim(0, 50)

# 显示网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 展示图形
plt.show()