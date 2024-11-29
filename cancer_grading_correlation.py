import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

def scatter_plot(df, x_col, y_col, category_col):
    """
    Scatter plot with color representing different categories.

    Parameters:
    - df: DataFrame containing the data
    - x_col: Column name for x-axis values
    - y_col: Column name for y-axis values
    - category_col: Column name for categories to color-code
    """
    plt.figure(figsize=(10, 6))
    unique_categories = df[category_col].cat.categories
    colors = sns.color_palette('husl', len(unique_categories))

    for i, category in enumerate(unique_categories):
        subset = df[df[category_col] == category]
        plt.scatter(subset[x_col].cat.codes, subset[y_col], color=colors[i], label=category, alpha=0.6)

    plt.xticks(ticks=range(len(unique_categories)), labels=unique_categories)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{y_col} vs {x_col} with {category_col} color coding')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def density_plot_2d(df, x_col, y_col):
    """
    2D density plot for two continuous variables.

    Parameters:
    - df: DataFrame containing the data
    - x_col: Column name for x-axis values
    - y_col: Column name for y-axis values
    """
    # Sort cancer grades for x-axis
    df[x_col] = pd.Categorical(df[x_col], categories=cancer_grade_order, ordered=True)
    x = df[x_col].cat.codes
    y = df[y_col]

    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=density, s=50, cmap='viridis', edgecolor='k')
    plt.xticks(ticks=range(len(cancer_grade_order)), labels=cancer_grade_order)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'2D Density Plot of {y_col} vs {x_col}')
    plt.colorbar(label='Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    average_results_df = pd.read_csv('../Gentl/glcm_average_gentl_results/correlation_average_generation_results_100_rois.csv')
    cancer_grades_df = pd.read_csv(
        '../Gentl/data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv')
    cancer_grades_df.rename(columns={'Anonymized ID': 'patient_id'}, inplace=True)
    cancer_grades_df.rename(columns={'Final Path': 'cancer_grade'}, inplace=True)
    # Step 3: Select only 'patient_id' and 'cancer_grade' columns to merge
    cancer_grades_df = cancer_grades_df[['patient_id', 'cancer_grade']]
    # Step 4: Merge the two dataframes on 'patient_id'
    merged_df = average_results_df.merge(cancer_grades_df, on='patient_id', how='left')
    # Cancer grade order for sorting
    cancer_grade_order = ['T0', 'Ta', 'Tis', 'T1', 'T2', 'T3', 'T4']
    # Set the order for cancer grades
    merged_df['cancer_grade'] = pd.Categorical(merged_df['cancer_grade'], categories=cancer_grade_order, ordered=True)

    # Automatically determine x_col and y_col based on available columns
    if 'average_distance' in merged_df.columns:
        x_col = 'cancer_grade'
        y_col = 'average_distance'
    elif 'average_generation' in merged_df.columns:
        x_col = 'cancer_grade'
        y_col = 'average_generation'
    else:
        raise ValueError("Neither 'average_distance' nor 'average_generation' found in the dataframe.")

    scatter_plot(merged_df, x_col, y_col, 'cancer_grade')
    density_plot_2d(merged_df, x_col, y_col)



