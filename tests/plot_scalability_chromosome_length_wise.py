import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import itertools
# import pingouin as pg
# from statannotations.Annotator import Annotator
# from decimal import Decimal
# from scipy.stats import variation
# import numpy as np
sns.set_theme(style="whitegrid")

layouts = [
    ["time"],
    ["generations"]
    
]

df=pd.read_csv("scalability_cr_len_iters_df.csv")

fig, axes = plt.subplot_mosaic(layouts, figsize=(22,12.5))

palette=None

# palette={"pearson":(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
          # "spearman": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
          # "kendall": (0.86, 0.3712, 0.33999999999999997)}

# #################################### Fig 2A healthy, overlap: ###############################################

# pairs=
# pvals_=
# pvals=[]
# for p_ in pvals_:
#     if p_<0.05:
#         pvals.append('%.2E' % Decimal(p_))
#     else:
#         pvals.append('$ns$')
# pvals=['{:.2f}'.format(x) for x in pvals]
xticklabels=list(df.columns)
# pairs_pvals_dict=dict(zip(pairs, pvals))
if palette:
    args = dict(data=df, palette=palette)
else:
    args = dict(data=df)
fig_, axes_ = plt.subplots(figsize=(20,10))
sns.set(font_scale = 1.2)
sns.set_style("white")
ax = sns.boxplot(**args)
# annot = Annotator(ax, pairs, **args)
# annot.set_custom_annotations(pvals)
# annot.annotate()
# plt.close()
# pairs_corrected=[]
# pvals_corrected=[]
# for j in range(len(annot.__dict__['annotations'])):
#     pair_1=annot.__dict__['annotations'][j].__dict__['structs'][0]['group']
#     pair_2=annot.__dict__['annotations'][j].__dict__['structs'][1]['group']
#     pairs_corrected.append((pair_1, pair_2))
#     pvals_corrected.append(pairs_pvals_dict[(pair_1, pair_2)])
# sns.set(font_scale = 1.2)
# sns.set_style("white")
ax = sns.boxplot(ax=axes["time"], **args)
title_prefix=None
plot_title=r'$\bf{A}$'
if title_prefix:
    if plot_title:
        plot_title=r"$\bf{" + title_prefix + "}$" + plot_title
    else:
        plot_title=r"$\bf{" + title_prefix + "}$"
# annot = Annotator(ax, pairs, **args)
# annot.configure(text_format='simple', loc='inside', verbose=2, fontsize=22.5) # fontsize=25
# annot.set_custom_annotations(pvals_corrected)
# annot.annotate()
title_loc='left'
ax.set_title(plot_title, loc=title_loc, fontsize=26) # fontsize=25, pad=20
xlabel='Chromosome length'
ax.set_xlabel(xlabel, fontsize=26) # fontsize=25, labelpad=20
ylabel='# of iterations'
ax.set_ylabel(ylabel, fontsize=26, labelpad=5) # fontsize=25, labelpad=20
yticks=[]
for j in ax.get_yticks():
    yticks.append(j)
ax.tick_params(right=True, labelright=True, left=False, labelleft=False, rotation=0)
ax.patch.set_edgecolor('black')
ax.grid(False)

plt.subplots_adjust(hspace=0.3)
plt.savefig('Fig2_cv_overlap_and_auroc.pdf', format='pdf', bbox_inches='tight')
plt.show()
