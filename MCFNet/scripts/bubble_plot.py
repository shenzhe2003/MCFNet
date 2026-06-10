import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np

colors = [
    'navy', 'teal', 'gold', 'darkorange', 'firebrick', 'orchid', 'indigo',
    'deepskyblue', 'forestgreen', 'mediumpurple', 'tomato', 'mediumvioletred',
    'turquoise',
    # 👇 倒数第五个位置新增的两种颜色
    'peru',
    'crimson', 'dodgerblue', 'olive', 'slategray', 'darkgreen', 'red'
]

# 设置最小气泡面积（单位：points²）
min_area = 64  # 对应直径约9pt

# 修正后的气泡大小
s_params = [max(p * 50, min_area) for p in params]
s_flops = [max(f * 50, min_area) for f in flops]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 左图：IoU vs FPS
axes[0].scatter(fps, iou, s=s_params, c=colors, alpha=0.7)
axes[0].set_xlabel('FPS ↑', fontsize=12)
axes[0].set_ylabel('IoU (%) ↑', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.5)

# 右图：Pd vs Inference Time
axes[1].scatter(time_ms, pd, s=s_flops, c=colors, alpha=0.7)
axes[1].set_xlabel('Inference Time (ms) ↓', fontsize=12)
axes[1].set_ylabel('$P_d$ (%) ↑', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.5)

# 图例：气泡大小
legend_elements0 = [
    Line2D([0], [0], marker='o', color='w', label='5M', markersize=np.sqrt(max(5 * 50, min_area)), markerfacecolor='gray', alpha=0.7),
    Line2D([0], [0], marker='o', color='w', label='10M', markersize=np.sqrt(max(10 * 50, min_area)), markerfacecolor='gray', alpha=0.7),
    Line2D([0], [0], marker='o', color='w', label='50M', markersize=np.sqrt(max(50 * 50, min_area)), markerfacecolor='gray', alpha=0.7)
]
legend_elements1 = [
    Line2D([0], [0], marker='o', color='w', label='5G', markersize=np.sqrt(max(5 * 50, min_area)), markerfacecolor='gray', alpha=0.7),
    Line2D([0], [0], marker='o', color='w', label='10G', markersize=np.sqrt(max(10 * 50, min_area)), markerfacecolor='gray', alpha=0.7),
    Line2D([0], [0], marker='o', color='w', label='50G', markersize=np.sqrt(max(50 * 50, min_area)), markerfacecolor='gray', alpha=0.7)
]

# 模型颜色图例
model_legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=model, markerfacecolor=color, markersize=10)
    for model, color in zip(models, colors)
]

# 图例添加
legend1_1 = axes[0].legend(
    handles=legend_elements0,
    loc='upper left',
    bbox_to_anchor=(0.35, 0.15),
    title='Bubble Size (Params)',
    title_fontsize=12,
    fontsize=11,
    borderaxespad=0.1,
    handletextpad=1.2,
    labelspacing=2.0,
    ncols=3,
    borderpad=1.2,
    frameon=True
)
axes[0].add_artist(legend1_1)
axes[0].legend(
    handles=model_legend_elements,
    loc='lower right',
    bbox_to_anchor=(1.0, 0.25),  # 上移的关键！默认是 (1.0, 0.0)
    title='Models',
    fontsize=8,
    title_fontsize=10
)


legend2_1 = axes[1].legend(
    handles=legend_elements1,
    loc='upper left',
    bbox_to_anchor=(0.35, 0.15),
    title='Bubble Size (FLOPs)',
    title_fontsize=12,
    fontsize=11,
    borderaxespad=0.1,
    handletextpad=1.2,
    ncols=3,
    labelspacing=2.0,
    borderpad=1.2,
    frameon=True
)
axes[1].add_artist(legend2_1)
axes[1].legend(
    handles=model_legend_elements,
    loc='lower right',
    title='Models',
    fontsize=8,
    title_fontsize=10
)

plt.tight_layout()
plt.savefig('bubble.pdf', format='pdf', bbox_inches='tight')
plt.close(fig)
