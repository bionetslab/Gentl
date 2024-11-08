import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pingouin as pg
from statannotations.Annotator import Annotator
from decimal import Decimal
from scipy.stats import variation
import numpy as np
sns.set_theme(style="whitegrid")
gt_legend_notation={'collectri_direct_all': 'CollecTRI (direct)', 'collectri_indirect_all': 'CollecTRI\n(co-regulation)', 'hippie': 'HIPPIE'}


cancer_types=['bladder urothelial carcinoma', 'brain lower grade glioma', 
              'breast invasive carcinoma', 'cervical & endocervical cancer', 
              'colon adenocarcinoma', 'esophageal carcinoma', 
              'kidney clear cell carcinoma', 'liver hepatocellular carcinoma', 
              'lung adenocarcinoma', 'pancreatic adenocarcinoma', 
              'prostate adenocarcinoma', 'stomach adenocarcinoma', 
              'testicular germ cell tumor', 'thymoma', 
              'uterine corpus endometrioid carcinoma']
cancer_type_abbreviations = list(range(1, len(cancer_types)+1))
cancer_types_lookup=dict(zip(cancer_types, cancer_type_abbreviations))
cancer_types_lookup_inverse=dict(zip(cancer_type_abbreviations, cancer_types))
cancer_types_str=str(cancer_types_lookup_inverse)
text_list = list(cancer_types_str[1:-1].split(","))
textstr = '\n'.join(text_list)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


cancer_tissue_folder_dir='../../results/cancer/'
healthy_tissue_folder_dir='../../results/healthy_tissue/'
healthy_tissue_mixed_folder_dir='../../results/healthy_tissue_mixed/'

folder_dirs=[cancer_tissue_folder_dir, healthy_tissue_folder_dir]

healthy_tissue_availability=pd.read_csv('../../healthy_tissue_availability.csv', index_col=0)


ground_truth_types = [
    ["collectri_direct_all"],
    ["collectri_indirect_all"],
    ["hippie"]
]

list_of_cancer_tissue=[]

_df_cancer=[]
_df_healthy=[]
df_mean_cancer=[]
df_mean_healthy=[]
for cancer_type in cancer_types:
    healthy_tissue_availability_per_cancer_type=healthy_tissue_availability.loc[cancer_type.lower()].to_dict()
    if healthy_tissue_availability_per_cancer_type['healthy_tissue_available']=='Yes':
        list_of_cancer_tissue.append(cancer_type)
        for ground_truth_type in ground_truth_types:
            _df_=pd.read_csv(f'../../results/cancer/{cancer_type}_{ground_truth_type[0]}.csv')
            _df_healthy_=pd.read_csv(f'../../results/healthy_tissue/{cancer_type}_{ground_truth_type[0]}.csv')
            
            # df_correlation_filtered=_df_[_df_['correlation_type']==correlation_groundtruth_dict[ground_truth_type[0]]]
            df_correlation_filtered=_df_.copy()
            # df_healthy_correlation_filtered=_df_healthy_[_df_healthy_['correlation_type']==correlation_groundtruth_dict[ground_truth_type[0]]]
            df_healthy_correlation_filtered=_df_healthy_.copy()
            
            mean_auroc=variation(df_correlation_filtered.loc[:, 'auroc'])
            mean_auprc=variation(df_correlation_filtered.loc[:, 'auprc'])
            mean_overlap=variation(df_correlation_filtered.loc[:, 'overlap_top_1000'])
            
            mean_auroc_healthy=variation(df_healthy_correlation_filtered.loc[:, 'auroc'])
            mean_auprc_healthy=variation(df_healthy_correlation_filtered.loc[:, 'auprc'])
            mean_overlap_healthy=variation(df_healthy_correlation_filtered.loc[:, 'overlap_top_1000'])
            
            _df_mean_cancer_=pd.DataFrame()
            _df_mean_cancer_['cancer_type']=[list(_df_['cancer_type'])[0]]
            _df_mean_cancer_['ground_truth_type']=[list(_df_['ground_truth_type'])[0]]
            _df_mean_cancer_['ground_truth_size']=[list(_df_['ground_truth_size'])[0]]
            _df_mean_cancer_['mean_auroc']=[mean_auroc]
            _df_mean_cancer_['mean_auprc']=[mean_auprc]
            _df_mean_cancer_['mean_overlap']=[mean_overlap]
            _df_mean_cancer_['correlation_type']=[list(_df_['correlation_type'])[0]]
            
            _df_mean_healthy_=pd.DataFrame()
            _df_mean_healthy_['cancer_type']=[list(_df_healthy_['cancer_type'])[0]]
            _df_mean_healthy_['ground_truth_type']=[list(_df_healthy_['ground_truth_type'])[0]]
            _df_mean_healthy_['ground_truth_size']=[list(_df_healthy_['ground_truth_size'])[0]]
            _df_mean_healthy_['mean_auroc']=[mean_auroc_healthy]
            _df_mean_healthy_['mean_auprc']=[mean_auprc_healthy]
            _df_mean_healthy_['mean_overlap']=[mean_overlap_healthy]
            _df_mean_healthy_['correlation_type']=[list(_df_healthy_['correlation_type'])[0]]
            
            _df_cancer.append(_df_)
            _df_healthy.append(_df_healthy_)
            df_mean_cancer.append(_df_mean_cancer_)
            df_mean_healthy.append(_df_mean_healthy_)
    else:
        raise Exception('Sorry, no healthy tissue expression found corresponding to this particular tissue type!')
    
df_cancer=pd.concat(_df_cancer, axis=0)
df_healthy=pd.concat(_df_healthy, axis=0)
df_mean_cancer=pd.concat(df_mean_cancer, axis=0)
df_mean_healthy=pd.concat(df_mean_healthy, axis=0)

df_cancer['cancer_type_abbreviation']=[cancer_types_lookup.get(item, item)  for item in list(df_cancer.cancer_type)]

df_mean_cancer_collectri_direct_all=df_mean_cancer[df_mean_cancer['ground_truth_type']=='collectri_direct_all']
df_mean_cancer_collectri_direct_all_auroc=df_mean_cancer_collectri_direct_all[['cancer_type','mean_auroc']].rename(columns={'mean_auroc': 'collectri_direct_all'})
df_mean_cancer_collectri_direct_all_auprc=df_mean_cancer_collectri_direct_all[['cancer_type','mean_auprc']].rename(columns={'mean_auprc': 'collectri_direct_all'})
df_mean_cancer_collectri_direct_all_overlap=df_mean_cancer_collectri_direct_all[['cancer_type','mean_overlap']].rename(columns={'mean_overlap': 'collectri_direct_all'})

df_mean_cancer_collectri_indirect_all=df_mean_cancer[df_mean_cancer['ground_truth_type']=='collectri_indirect_all']
df_mean_cancer_collectri_indirect_all_auroc=df_mean_cancer_collectri_indirect_all[['mean_auroc']].rename(columns={'mean_auroc': 'collectri_indirect_all'})
df_mean_cancer_collectri_indirect_all_auprc=df_mean_cancer_collectri_indirect_all[['mean_auprc']].rename(columns={'mean_auprc': 'collectri_indirect_all'})
df_mean_cancer_collectri_indirect_all_overlap=df_mean_cancer_collectri_indirect_all[['mean_overlap']].rename(columns={'mean_overlap': 'collectri_indirect_all'})

df_mean_cancer_hippie=df_mean_cancer[df_mean_cancer['ground_truth_type']=='hippie']
df_mean_cancer_hippie_auroc=df_mean_cancer_hippie[['mean_auroc']].rename(columns={'mean_auroc': 'hippie'})
df_mean_cancer_hippie_auprc=df_mean_cancer_hippie[['mean_auprc']].rename(columns={'mean_auprc': 'hippie'})
df_mean_cancer_hippie_overlap=df_mean_cancer_hippie[['mean_overlap']].rename(columns={'mean_overlap': 'hippie'})

df_mean_cancer_auroc=pd.concat([df_mean_cancer_collectri_direct_all_auroc,df_mean_cancer_collectri_indirect_all_auroc,df_mean_cancer_hippie_auroc], axis=1)
df_mean_cancer_auroc.set_index(['cancer_type'], inplace=True)

df_mean_cancer_overlap=pd.concat([df_mean_cancer_collectri_direct_all_overlap,df_mean_cancer_collectri_indirect_all_overlap,df_mean_cancer_hippie_overlap], axis=1)
df_mean_cancer_overlap.set_index(['cancer_type'], inplace=True)

# --------------------------------------------------------------------------------

list_of_healthy_tissue=[]

_df_cancer=[]
_df_healthy=[]
df_mean_cancer=[]
df_mean_healthy=[]
for cancer_type in cancer_types:
    healthy_tissue_availability_per_cancer_type=healthy_tissue_availability.loc[cancer_type.lower()].to_dict()
    if healthy_tissue_availability_per_cancer_type['healthy_tissue_available']=='Yes':
        if not(healthy_tissue_availability_per_cancer_type["organ_name"] in list_of_healthy_tissue):
            list_of_healthy_tissue.append(healthy_tissue_availability_per_cancer_type["organ_name"])
            for ground_truth_type in ground_truth_types:
                _df_=pd.read_csv(f'../../results/cancer/{cancer_type}_{ground_truth_type[0]}.csv')
                _df_healthy_=pd.read_csv(f'../../results/healthy_tissue/{cancer_type}_{ground_truth_type[0]}.csv')
                
                # df_correlation_filtered=_df_[_df_['correlation_type']==correlation_groundtruth_dict[ground_truth_type[0]]]
                df_correlation_filtered=_df_.copy()
                # df_healthy_correlation_filtered=_df_healthy_[_df_healthy_['correlation_type']==correlation_groundtruth_dict[ground_truth_type[0]]]
                df_healthy_correlation_filtered=_df_healthy_.copy()
                
                mean_auroc=[variation(df_correlation_filtered[df_correlation_filtered['correlation_type']=='pearson'].loc[:, 'auroc']), variation(df_correlation_filtered[df_correlation_filtered['correlation_type']=='spearman'].loc[:, 'auroc']), variation(df_correlation_filtered[df_correlation_filtered['correlation_type']=='kendall'].loc[:, 'auroc'])]
                mean_auprc=[variation(df_correlation_filtered[df_correlation_filtered['correlation_type']=='pearson'].loc[:, 'auprc']), variation(df_correlation_filtered[df_correlation_filtered['correlation_type']=='spearman'].loc[:, 'auprc']), variation(df_correlation_filtered[df_correlation_filtered['correlation_type']=='kendall'].loc[:, 'auprc'])]
                mean_overlap=[variation(df_correlation_filtered[df_correlation_filtered['correlation_type']=='pearson'].loc[:, 'overlap_top_1000']), variation(df_correlation_filtered[df_correlation_filtered['correlation_type']=='spearman'].loc[:, 'overlap_top_1000']), variation(df_correlation_filtered[df_correlation_filtered['correlation_type']=='kendall'].loc[:, 'overlap_top_1000'])]
                
                mean_auroc_healthy=[variation(df_healthy_correlation_filtered[df_healthy_correlation_filtered['correlation_type']=='pearson'].loc[:, 'auroc']), variation(df_healthy_correlation_filtered[df_healthy_correlation_filtered['correlation_type']=='spearman'].loc[:, 'auroc']), variation(df_healthy_correlation_filtered[df_healthy_correlation_filtered['correlation_type']=='kendall'].loc[:, 'auroc'])]
                mean_auprc_healthy=[variation(df_healthy_correlation_filtered[df_healthy_correlation_filtered['correlation_type']=='pearson'].loc[:, 'auprc']), variation(df_healthy_correlation_filtered[df_healthy_correlation_filtered['correlation_type']=='spearman'].loc[:, 'auprc']), variation(df_healthy_correlation_filtered[df_healthy_correlation_filtered['correlation_type']=='kendall'].loc[:, 'auprc'])]
                mean_overlap_healthy=[variation(df_healthy_correlation_filtered[df_healthy_correlation_filtered['correlation_type']=='pearson'].loc[:, 'overlap_top_1000']), variation(df_healthy_correlation_filtered[df_healthy_correlation_filtered['correlation_type']=='spearman'].loc[:, 'overlap_top_1000']), variation(df_healthy_correlation_filtered[df_healthy_correlation_filtered['correlation_type']=='kendall'].loc[:, 'overlap_top_1000'])]
                
                
                _df_mean_cancer_=pd.DataFrame()
                _df_mean_cancer_['cancer_type']=[list(_df_['cancer_type'])[0]]*3
                _df_mean_cancer_['ground_truth_type']=[list(_df_['ground_truth_type'])[0]]*3
                _df_mean_cancer_['ground_truth_size']=[list(_df_['ground_truth_size'])[0]]*3
                _df_mean_cancer_['mean_auroc']=mean_auroc
                _df_mean_cancer_['mean_auprc']=mean_auprc
                _df_mean_cancer_['mean_overlap']=mean_overlap
                _df_mean_cancer_['correlation_type']=['pearson', 'spearman', 'kendall']
                
                _df_mean_healthy_=pd.DataFrame()
                _df_mean_healthy_['cancer_type']=[list(_df_healthy_['cancer_type'])[0]]*3
                _df_mean_healthy_['ground_truth_type']=[list(_df_healthy_['ground_truth_type'])[0]]*3
                _df_mean_healthy_['ground_truth_size']=[list(_df_healthy_['ground_truth_size'])[0]]*3
                _df_mean_healthy_['mean_auroc']=mean_auroc_healthy
                _df_mean_healthy_['mean_auprc']=mean_auprc_healthy
                _df_mean_healthy_['mean_overlap']=mean_overlap_healthy
                _df_mean_healthy_['correlation_type']=['pearson', 'spearman', 'kendall']
                
                _df_cancer.append(_df_)
                _df_healthy.append(_df_healthy_)
                df_mean_cancer.append(_df_mean_cancer_)
                df_mean_healthy.append(_df_mean_healthy_)
    else:
        raise Exception('Sorry, no healthy tissue expression found corresponding to this particular tissue type!')
    
df_cancer=pd.concat(_df_cancer, axis=0)
df_healthy=pd.concat(_df_healthy, axis=0)
df_mean_cancer=pd.concat(df_mean_cancer, axis=0)
df_mean_healthy=pd.concat(df_mean_healthy, axis=0)

df_cancer['cancer_type_abbreviation']=[cancer_types_lookup.get(item, item)  for item in list(df_cancer.cancer_type)]

df_healthy_and_cancer=pd.concat([df_healthy, df_cancer], axis=0)
df_healthy_and_cancer=df_healthy_and_cancer[df_healthy_and_cancer['correlation_type']=='pearson']
df_healthy_and_cancer=df_healthy_and_cancer[df_healthy_and_cancer['ground_truth_type']=='collectri_direct_all']
cv_overlap_median=np.median(list(df_healthy_and_cancer.overlap_top_1000))
cv_overlap_mean=np.mean(list(df_healthy_and_cancer.overlap_top_1000))

fig2_plot_layouts = [
    ["healthy_overlap", "healthy_auroc"],
    ["cancer_overlap", "cancer_auroc"]
    
]

fig2, axes2 = plt.subplot_mosaic(fig2_plot_layouts, figsize=(22,12.5))

palette={"pearson":(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
          "spearman": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
          "kendall": (0.86, 0.3712, 0.33999999999999997)}

# #################################### Fig 2A healthy, overlap: ###############################################

df_healthy=df_mean_healthy.copy()

correlation_types=['pearson', 'spearman', 'kendall']
correlation_type_combinations=list(itertools.combinations(correlation_types, 2))

mwu_auroc=[]
mwu_overlap=[]
mwu_auprc=[]
for ground_truth_type in ground_truth_types:
    df_ground_truth_type=df_healthy[df_healthy['ground_truth_type']==ground_truth_type[0]]
    for k in correlation_type_combinations:
        x_auroc=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[0]][['mean_auroc']]
        y_auroc=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[1]][['mean_auroc']]
        df_mwu_auroc=pg.mwu(x_auroc, y_auroc, alternative='two-sided')[['U-val', 'p-val']]
        df_mwu_auroc=df_mwu_auroc.rename(columns={'U-val': 'U_auroc', 'p-val': 'p_auroc'})
        df_mwu_auroc.index=[((ground_truth_type[0], k[0]), (ground_truth_type[0], k[1]))]
        mwu_auroc.append(df_mwu_auroc)
        # ---
        x_overlap=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[0]][['mean_overlap']]
        y_overlap=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[1]][['mean_overlap']]
        df_mwu_overlap=pg.mwu(x_overlap, y_overlap, alternative='two-sided')[['U-val', 'p-val']]
        df_mwu_overlap=df_mwu_overlap.rename(columns={'U-val': 'U_overlap', 'p-val': 'p_overlap'})
        df_mwu_overlap.index=[((ground_truth_type[0], k[0]), (ground_truth_type[0], k[1]))]
        mwu_overlap.append(df_mwu_overlap)
        # ---
        x_auprc=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[0]][['mean_auprc']]
        y_auprc=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[1]][['mean_auprc']]
        df_mwu_auprc=pg.mwu(x_auprc, y_auprc, alternative='two-sided')[['U-val', 'p-val']]
        df_mwu_auprc=df_mwu_auprc.rename(columns={'U-val': 'U_auprc', 'p-val': 'p_auprc'})
        df_mwu_auprc.index=[((ground_truth_type[0], k[0]), (ground_truth_type[0], k[1]))]
        mwu_auprc.append(df_mwu_auprc)
mwu_auroc_healthy=pd.concat(mwu_auroc, axis=0)
mwu_overlap_healthy=pd.concat(mwu_overlap, axis=0)
mwu_auprc_healthy=pd.concat(mwu_auprc, axis=0)


df_plot = df_healthy.copy()
pairs=list(mwu_overlap_healthy.index)
pvals_=list(mwu_overlap_healthy.p_overlap)
pvals=[]
for p_ in pvals_:
    if p_<0.05:
        pvals.append('%.2E' % Decimal(p_))
    else:
        pvals.append('$ns$')
# pvals=['{:.2f}'.format(x) for x in pvals]
xticklabels=['CollecTRI\n(direct)', 'CollecTRI\n(co-regulation)', 'HIPPIE']
pairs_pvals_dict=dict(zip(pairs, pvals))
if palette:
    args = dict(x="ground_truth_type", y="mean_overlap", data=df_plot, hue="correlation_type", hue_order=['pearson','spearman', 'kendall'], order=['collectri_direct_all', 'collectri_indirect_all', 'hippie'], palette=palette)
else:
    args = dict(x="ground_truth_type", y="mean_overlap", data=df_plot, hue="correlation_type", hue_order=['pearson','spearman', 'kendall'], order=['collectri_direct_all', 'collectri_indirect_all', 'hippie'])
fig_, axes_ = plt.subplots(figsize=(20,10))
sns.set(font_scale = 1.2)
sns.set_style("white")
ax = sns.boxplot(**args)
annot = Annotator(ax, pairs, **args)
annot.set_custom_annotations(pvals)
annot.annotate()
plt.close()
pairs_corrected=[]
pvals_corrected=[]
for j in range(len(annot.__dict__['annotations'])):
    pair_1=annot.__dict__['annotations'][j].__dict__['structs'][0]['group']
    pair_2=annot.__dict__['annotations'][j].__dict__['structs'][1]['group']   
    pairs_corrected.append((pair_1, pair_2))
    pvals_corrected.append(pairs_pvals_dict[(pair_1, pair_2)])
sns.set(font_scale = 1.2)
sns.set_style("white")
ax = sns.boxplot(ax=axes2["healthy_overlap"], **args) 
title_prefix=None
plot_title=r'$\bf{A}$'
if title_prefix:
    if plot_title:
        plot_title=r"$\bf{" + title_prefix + "}$" + plot_title
    else:
        plot_title=r"$\bf{" + title_prefix + "}$"
annot = Annotator(ax, pairs, **args)
annot.configure(text_format='simple', loc='inside', verbose=2, fontsize=22.5) # fontsize=25
annot.set_custom_annotations(pvals_corrected)
annot.annotate()
title_loc='left'
ax.set_title(plot_title, loc=title_loc, fontsize=26) # fontsize=25, pad=20
xlabel=''
ax.set_xlabel(xlabel, fontsize=26) # fontsize=25, labelpad=20
ylabel='O1K'
ax.set_ylabel(ylabel, fontsize=26, labelpad=5) # fontsize=25, labelpad=20
ax.set_xticklabels(xticklabels, size=22.5) # size=20
yticks=[]
for j in ax.get_yticks():
    yticks.append(round(j,1))
ax.set_yticklabels(['{:.0f}'.format(t._y) for t in ax.get_yticklabels()], size=19)
ax.tick_params(right=True, labelright=True, left=False, labelleft=False, rotation=0)
for lh in ax.get_legend().get_patches(): 
    lh.set_alpha(0.2)
hands, labs = ax.get_legend_handles_labels()
ax.legend(handles=hands, title=None, loc='upper right', labels=['$r_P$', '$r_S$', r'$\tau$'], borderaxespad=0.5, fontsize=25, title_fontsize=25, frameon=True, markerscale=3)
# ax.legend(title=None, loc='upper right', labels=['$r_P$', '$r_S$', '$\tau$'], borderaxespad=0.5, fontsize=25, title_fontsize=25, frameon=True, markerscale=3)
# sns.move_legend(ax, "upper right", borderaxespad=0.5, fontsize=25, title_fontsize=25, frameon=True, markerscale=3, labels=['$r_P$', '$r_S$', '$\tau$'])
ax.patch.set_edgecolor('black')
ax.grid(False)


# #################################### Fig 2B healthy, AUROC: ###############################################

df_plot = df_healthy.copy()
pairs=list(mwu_auroc_healthy.index)
pvals_=list(mwu_auroc_healthy.p_auroc)
pvals=[]
for p_ in pvals_:
    if p_<0.05:
        pvals.append('%.2E' % Decimal(p_))
    else:
        pvals.append('$ns$')
# pvals=['{:.2f}'.format(x) for x in pvals]
xticklabels=['CollecTRI\n(direct)', 'CollecTRI\n(co-regulation)', 'HIPPIE']
pairs_pvals_dict=dict(zip(pairs, pvals))
if palette:
    args = dict(x="ground_truth_type", y="mean_auroc", data=df_plot, hue="correlation_type", hue_order=['pearson','spearman', 'kendall'], order=['collectri_direct_all', 'collectri_indirect_all', 'hippie'], palette=palette)
else:
    args = dict(x="ground_truth_type", y="mean_auroc", data=df_plot, hue="correlation_type", hue_order=['pearson','spearman', 'kendall'], order=['collectri_direct_all', 'collectri_indirect_all', 'hippie'])
fig_, axes_ = plt.subplots(figsize=(20,10))
sns.set(font_scale = 1.2)
sns.set_style("white")
ax = sns.boxplot(**args) 
annot = Annotator(ax, pairs, **args)
annot.set_custom_annotations(pvals)
annot.annotate()
plt.close()
pairs_corrected=[]
pvals_corrected=[]
for j in range(len(annot.__dict__['annotations'])):
    pair_1=annot.__dict__['annotations'][j].__dict__['structs'][0]['group']
    pair_2=annot.__dict__['annotations'][j].__dict__['structs'][1]['group']   
    pairs_corrected.append((pair_1, pair_2))
    pvals_corrected.append(pairs_pvals_dict[(pair_1, pair_2)])
sns.set(font_scale = 1.2)
sns.set_style("white")
ax = sns.boxplot(ax=axes2["healthy_auroc"], **args) 
title_prefix=None
plot_title=r'$\bf{B}$'
if title_prefix:
    if plot_title:
        plot_title=r"$\bf{" + title_prefix + "}$" + plot_title
    else:
        plot_title=r"$\bf{" + title_prefix + "}$"
annot = Annotator(ax, pairs, **args)
annot.configure(text_format='simple', loc='inside', verbose=2, fontsize=22.5) # fontsize=25
annot.set_custom_annotations(pvals_corrected)
annot.annotate()
title_loc='left'
ax.set_title(plot_title, loc=title_loc, fontsize=26) # fontsize=25, pad=20
xlabel=''
ax.set_xlabel(xlabel, fontsize=26) # fontsize=25, labelpad=20
ylabel='AUROC'
ax.set_ylabel(ylabel, fontsize=26, labelpad=5) # fontsize=25, labelpad=20
ax.set_xticklabels(xticklabels, size=22.5) # size=20
ax.get_legend().set_visible(False)
yticks=[]
for j in ax.get_yticks():
    yticks.append(round(j,1))
ax.set_yticklabels(['{:.3f}'.format(t._y) for t in ax.get_yticklabels()], size=19)
ax.tick_params(right=True, labelright=True, left=False, labelleft=False, rotation=0)
# setp(ax, frame_on=False)
ax.patch.set_edgecolor('black')
ax.grid(False)


# #################################### Fig 2C cancer, overlap: ###############################################

df_cancer=df_mean_cancer.copy()

correlation_types=['pearson', 'spearman', 'kendall']
correlation_type_combinations=list(itertools.combinations(correlation_types, 2))

mwu_auroc=[]
mwu_overlap=[]
mwu_auprc=[]
for ground_truth_type in ground_truth_types:
    df_ground_truth_type=df_cancer[df_cancer['ground_truth_type']==ground_truth_type[0]]
    for k in correlation_type_combinations:
        x_auroc=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[0]][['mean_auroc']]
        y_auroc=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[1]][['mean_auroc']]
        df_mwu_auroc=pg.mwu(x_auroc, y_auroc, alternative='two-sided')[['U-val', 'p-val']]
        df_mwu_auroc=df_mwu_auroc.rename(columns={'U-val': 'U_auroc', 'p-val': 'p_auroc'})
        df_mwu_auroc.index=[((ground_truth_type[0], k[0]), (ground_truth_type[0], k[1]))]
        mwu_auroc.append(df_mwu_auroc)
        # ---
        x_overlap=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[0]][['mean_overlap']]
        y_overlap=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[1]][['mean_overlap']]
        df_mwu_overlap=pg.mwu(x_overlap, y_overlap, alternative='two-sided')[['U-val', 'p-val']]
        df_mwu_overlap=df_mwu_overlap.rename(columns={'U-val': 'U_overlap', 'p-val': 'p_overlap'})
        df_mwu_overlap.index=[((ground_truth_type[0], k[0]), (ground_truth_type[0], k[1]))]
        mwu_overlap.append(df_mwu_overlap)
        # ---
        x_auprc=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[0]][['mean_auprc']]
        y_auprc=df_ground_truth_type[df_ground_truth_type['correlation_type']==k[1]][['mean_auprc']]
        df_mwu_auprc=pg.mwu(x_auprc, y_auprc, alternative='two-sided')[['U-val', 'p-val']]
        df_mwu_auprc=df_mwu_auprc.rename(columns={'U-val': 'U_auprc', 'p-val': 'p_auprc'})
        df_mwu_auprc.index=[((ground_truth_type[0], k[0]), (ground_truth_type[0], k[1]))]
        mwu_auprc.append(df_mwu_auprc)
mwu_auroc_cancer=pd.concat(mwu_auroc, axis=0)
mwu_overlap_cancer=pd.concat(mwu_overlap, axis=0)
mwu_auprc_cancer=pd.concat(mwu_auprc, axis=0)


df_plot = df_cancer.copy()
pairs=list(mwu_overlap_cancer.index)
pvals_=list(mwu_overlap_cancer.p_overlap)
pvals=[]
for p_ in pvals_:
    if p_<0.05:
        pvals.append('%.2E' % Decimal(p_))
    else:
        pvals.append('$ns$')
# pvals=['{:.2f}'.format(x) for x in pvals]
xticklabels=['CollecTRI\n(direct)', 'CollecTRI\n(co-regulation)', 'HIPPIE']
pairs_pvals_dict=dict(zip(pairs, pvals))
if palette:
    args = dict(x="ground_truth_type", y="mean_overlap", data=df_plot, hue="correlation_type", hue_order=['pearson','spearman', 'kendall'], order=['collectri_direct_all', 'collectri_indirect_all', 'hippie'], palette=palette)
else:
    args = dict(x="ground_truth_type", y="mean_overlap", data=df_plot, hue="correlation_type", hue_order=['pearson','spearman', 'kendall'], order=['collectri_direct_all', 'collectri_indirect_all', 'hippie'])
fig_, axes_ = plt.subplots(figsize=(20,10))
sns.set(font_scale = 1.2)
sns.set_style("white")
ax = sns.boxplot(**args) 
annot = Annotator(ax, pairs, **args)
annot.set_custom_annotations(pvals)
annot.annotate()
plt.close()
pairs_corrected=[]
pvals_corrected=[]
for j in range(len(annot.__dict__['annotations'])):
    pair_1=annot.__dict__['annotations'][j].__dict__['structs'][0]['group']
    pair_2=annot.__dict__['annotations'][j].__dict__['structs'][1]['group']   
    pairs_corrected.append((pair_1, pair_2))
    pvals_corrected.append(pairs_pvals_dict[(pair_1, pair_2)])
sns.set(font_scale = 1.2)
sns.set_style("white")
ax = sns.boxplot(ax=axes2["cancer_overlap"], **args) 
title_prefix=None
plot_title=r'$\bf{C}$'
if title_prefix:
    if plot_title:
        plot_title=r"$\bf{" + title_prefix + "}$" + plot_title
    else:
        plot_title=r"$\bf{" + title_prefix + "}$"
annot = Annotator(ax, pairs, **args)
annot.configure(text_format='simple', loc='inside', verbose=2, fontsize=22.5) # fontsize=25
annot.set_custom_annotations(pvals_corrected)
annot.annotate()
title_loc='left'
ax.set_title(plot_title, loc=title_loc, fontsize=26) # fontsize=25, pad=20
xlabel=''
ax.set_xlabel(xlabel, fontsize=26) # fontsize=25, labelpad=20
ylabel='O1K'
ax.set_ylabel(ylabel, fontsize=26, labelpad=5) # fontsize=25, labelpad=20
ax.set_xticklabels(xticklabels, size=22.5) # size=20
ax.get_legend().set_visible(False)
yticks=[]
for j in ax.get_yticks():
    yticks.append(round(j,1))
ax.set_yticklabels(['{:.0f}'.format(t._y) for t in ax.get_yticklabels()], size=19)
ax.tick_params(right=True, labelright=True, left=False, labelleft=False, rotation=0)
# setp(ax, frame_on=False)
ax.patch.set_edgecolor('black')
ax.grid(False)


# #################################### Fig 2D cancer, AUROC: ###############################################

df_plot = df_cancer.copy()
pairs=list(mwu_auroc_cancer.index)
pvals_=list(mwu_auroc_cancer.p_auroc)
pvals=[]
for p_ in pvals_:
    if p_<0.05:
        pvals.append('%.2E' % Decimal(p_))
    else:
        pvals.append('$ns$')
# pvals=['{:.2f}'.format(x) for x in pvals]
xticklabels=['CollecTRI\n(direct)', 'CollecTRI\n(co-regulation)', 'HIPPIE']
pairs_pvals_dict=dict(zip(pairs, pvals))
if palette:
    args = dict(x="ground_truth_type", y="mean_auroc", data=df_plot, hue="correlation_type", hue_order=['pearson','spearman', 'kendall'], order=['collectri_direct_all', 'collectri_indirect_all', 'hippie'], palette=palette)
else:
    args = dict(x="ground_truth_type", y="mean_auroc", data=df_plot, hue="correlation_type", hue_order=['pearson','spearman', 'kendall'], order=['collectri_direct_all', 'collectri_indirect_all', 'hippie'])
fig_, axes_ = plt.subplots(figsize=(20,10))
sns.set(font_scale = 1.2)
sns.set_style("white")
ax = sns.boxplot(**args) 
annot = Annotator(ax, pairs, **args)
annot.set_custom_annotations(pvals)
annot.annotate()
plt.close()
pairs_corrected=[]
pvals_corrected=[]
for j in range(len(annot.__dict__['annotations'])):
    pair_1=annot.__dict__['annotations'][j].__dict__['structs'][0]['group']
    pair_2=annot.__dict__['annotations'][j].__dict__['structs'][1]['group']   
    pairs_corrected.append((pair_1, pair_2))
    pvals_corrected.append(pairs_pvals_dict[(pair_1, pair_2)])
sns.set(font_scale = 1.2)
sns.set_style("white")
ax = sns.boxplot(ax=axes2["cancer_auroc"], **args) 
title_prefix=None
plot_title=r'$\bf{D}$'
if title_prefix:
    if plot_title:
        plot_title=r"$\bf{" + title_prefix + "}$" + plot_title
    else:
        plot_title=r"$\bf{" + title_prefix + "}$"
annot = Annotator(ax, pairs, **args)
annot.configure(text_format='simple', loc='inside', verbose=2, fontsize=22.5) # fontsize=25
annot.set_custom_annotations(pvals_corrected)
annot.annotate()
title_loc='left'
ax.set_title(plot_title, loc=title_loc, fontsize=26) # fontsize=25, pad=20
xlabel=''
ax.set_xlabel(xlabel, fontsize=26) # fontsize=25, labelpad=20
ylabel='AUROC'

ax.set_ylabel(ylabel, fontsize=26, labelpad=5) # fontsize=25, labelpad=20
ax.set_xticklabels(xticklabels, size=22.5) # size=20
ax.get_legend().set_visible(False)
ax.set_yticklabels(['{:.2f}'.format(t._y) for t in ax.get_yticklabels()], size=19) # size=20
ax.tick_params(right=True, labelright=True, left=False, labelleft=False, rotation=0)
# setp(ax, frame_on=False)
ax.patch.set_edgecolor('black')
ax.grid(False)


plt.subplots_adjust(hspace=0.3)
plt.savefig('Fig2_cv_overlap_and_auroc.pdf', format='pdf', bbox_inches='tight')
plt.show()
