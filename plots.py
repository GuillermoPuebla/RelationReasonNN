import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# Import tab20c colors
c = plt.get_cmap('tab20c').colors

# Groups (models)
N = 2

# Results per model (SG, Seq2seq)
gen_acc = [0.872, 0.954]
con_viol = [0.079, 0.996]
stats_viol = [0.216, 0.155]
shuffled = [0.654, 0.523]

# The x locations for the groups
ind = np.arange(N)

# The width of the bars
width = 0.2

# Add a subplot per result
fig, ax = plt.subplots(figsize=(8.2, 4.6))
rects1 = ax.bar(ind - width, gen_acc, width, color=c[0], edgecolor='0.0')
rects2 = ax.bar(ind, con_viol, width, color=c[14], edgecolor='0.0')
rects3 = ax.bar(ind + width, stats_viol, width, color=c[13], edgecolor='0.0')
rects4 = ax.bar(ind + width * 2, shuffled, width, color=c[15], edgecolor='0.0')

# Set axis label size
plt.tick_params(labelsize=12)

# Add y-axis grid
ax.set_yticks(np.arange(0, 1.1, step=0.1))
plt.grid(True, axis='y', linestyle='-')

# Don't allow the axis to be on top of your data
ax.set_axisbelow(True)

# Set y-axis limits
ax.set_ylim([0, 1.05])

# Add some text for labels, title and axes ticks
ax.set_ylabel('Proportion correct answers', fontsize=13)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Story Gestalt', 'Seq2seq+Attention'), fontsize=13)


# Legend
fig.legend((rects1[0], rects2[0], rects3[0], rects4[0]),
           ('Baseline', 'Concept violation', 'Correlation violation', 'Shuffled propositions'),
           loc='center right',
           fontsize=12,
           edgecolor='0.0')

plt.tight_layout()
fig.subplots_adjust(right=0.69)


# Write proportions on the top of the bars
def auto_label(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%s' % height,
                ha='center', va='bottom')


auto_label(rects1)
auto_label(rects2)
auto_label(rects3)
auto_label(rects4)

# Save plot
fig.savefig('rel_reason_results.pdf')
