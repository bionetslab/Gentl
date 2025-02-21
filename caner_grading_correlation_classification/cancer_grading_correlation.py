import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils import class_weight

# Global variable for cancer grade order
cancer_grade_order = ['T0', 'Ta', 'Tis', 'T1', 'T2', 'T3', 'T4']


def scatter_plot(df, x_col, y_col, category_col, y_label_override=None, title_override=None, save_path=None):
    """
    Creates a scatter plot with colors representing different categories.

    :param df: DataFrame containing the data.
    :param x_col: Column name for the x-axis.
    :param y_col: Column name for the y-axis.
    :param category_col: Column name for the categorical variable (used for color coding).
    :param y_label_override: Optional override for the y-axis label.
    :param title_override: Optional override for the plot title.
    """
    plt.figure(figsize=(10, 6))
    unique_categories = df[category_col].cat.categories
    colors = sns.color_palette('husl', len(unique_categories))

    for i, category in enumerate(unique_categories):
        subset = df[df[category_col] == category]
        plt.scatter(subset[x_col].cat.codes, subset[y_col], color=colors[i], label=category, alpha=0.6)

    plt.xticks(ticks=range(len(unique_categories)), labels=unique_categories)
    plt.xlabel(x_col)
    plt.ylabel(y_label_override if y_label_override is not None else y_col)
    plt.title(title_override if title_override is not None else f'{y_col} vs {x_col} with {category_col} color coding')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the figure as PDF if a path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()


def density_plot_2d(df, x_col, y_col, y_label_override=None, title_override=None, save_path=None):
    """
    Creates a 2D density plot for the data.

    :param df: DataFrame containing the data.
    :param x_col: Column name for the x-axis.
    :param y_col: Column name for the y-axis.
    :param y_label_override: Optional override for the y-axis label.
    :param title_override: Optional override for the plot title.
    """
    # Set the x-axis to use the ordered cancer grades
    df[x_col] = pd.Categorical(df[x_col], categories=cancer_grade_order, ordered=True)
    x = df[x_col].cat.codes
    y = df[y_col]

    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=density, s=50, cmap='viridis', edgecolor='k')
    plt.xticks(ticks=range(len(cancer_grade_order)), labels=cancer_grade_order)
    plt.xlabel(x_col)
    plt.ylabel(y_label_override if y_label_override is not None else y_col)
    plt.title(title_override if title_override is not None else f'2D Density Plot of {y_col} vs {x_col}')
    plt.colorbar(label='Density')
    plt.grid(True)
    plt.tight_layout()
    # Save the figure as PDF if a path is provided
    if save_path:
        plt.savefig(save_path)
    plt.show()


def evaluate_model(y_true, y_pred, model_name, feature_column):
    """
    Evaluates the model's performance and prints accuracy and F1-score.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param model_name: Name of the model.
    :param feature_column: Feature column used in the model.
    """
    print(f"\nEvaluating model: {model_name}")
    accuracy_val = accuracy_score(y_true, y_pred) * 100
    f1_score_val = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    print(f"Accuracy for MIBC vs NMIBC based on {feature_column}: {accuracy_val:.2f}%")
    print(f"F1-score for MIBC vs NMIBC based on {feature_column}: {f1_score_val:.2f}%")
    print(classification_report(y_true, y_pred, zero_division=0))


def decision_tree(merged_df, feature_column):
    """
    Runs a Decision Tree classifier and prints evaluation metrics.

    :param merged_df: Merged DataFrame containing features and labels.
    :param feature_column: Column name of the feature to use.
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        class_weight='balanced', criterion='entropy',
        max_depth=2, min_samples_leaf=7, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Decision Tree", feature_column)


def knn_classifier(merged_df, feature_column, n_neighbors=5):
    """
    Runs a K-Nearest Neighbors classifier and prints evaluation metrics.

    :param merged_df: Merged DataFrame containing features and labels.
    :param feature_column: Column name of the feature to use.
    :param n_neighbors: Number of neighbors (default=5).
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "KNN", feature_column)


def lda_classifier(merged_df, feature_column):
    """
    Runs a Linear Discriminant Analysis classifier and prints evaluation metrics.

    :param merged_df: Merged DataFrame containing features and labels.
    :param feature_column: Column name of the feature to use.
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "LDA", feature_column)


def logistic_regression(merged_df, feature_column):
    """
    Runs a Logistic Regression classifier and prints evaluation metrics.

    :param merged_df: Merged DataFrame containing features and labels.
    :param feature_column: Column name of the feature to use.
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Logistic Regression", feature_column)


def random_forest(merged_df, feature_column, n_estimators=100):
    """
    Runs a Random Forest classifier and prints evaluation metrics.

    :param merged_df: Merged DataFrame containing features and labels.
    :param feature_column: Column name of the feature to use.
    :param n_estimators: Number of trees in the forest (default=100).
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight_dict, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Random Forest", feature_column)


def svm_classifier(merged_df, feature_column):
    """
    Runs a Support Vector Machine (SVM) classifier and prints evaluation metrics.

    :param merged_df: Merged DataFrame containing features and labels.
    :param feature_column: Column name of the feature to use.
    """
    df_filtered = merged_df[merged_df['cancer_grade'] != 'T0'].copy()
    stage_mapping = {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
    df_filtered['cancer_stage'] = df_filtered['cancer_grade'].map(stage_mapping)
    if df_filtered['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = df_filtered[[feature_column]]
    y = df_filtered['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "SVM", feature_column)


# ===================== Main =====================
if __name__ == "__main__":
    # Step 1: Load files
    file_path = '../glcm_bladder_average_gentl_results/40/average_mean_absolute_distance_results/dissimilarity_average_mean_absolute_distance_results_40_rois.csv'
    average_results_df = pd.read_csv(file_path)
    cancer_grades_df = pd.read_csv(
        '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv')

    # Step 2: Rename columns and remove rows with missing values
    cancer_grades_df.rename(columns={'Anonymized ID': 'patient_id', 'Final Path': 'cancer_grade'}, inplace=True)
    cancer_grades_df.dropna(subset=['patient_id', 'cancer_grade'], inplace=True)
    cancer_grades_df = cancer_grades_df[['patient_id', 'cancer_grade']]

    # Step 3: Merge the two DataFrames on 'patient_id'
    merged_df = average_results_df.merge(cancer_grades_df, on='patient_id', how='left')

    # Set the order for cancer_grade
    merged_df['cancer_grade'] = pd.Categorical(merged_df['cancer_grade'], categories=cancer_grade_order, ordered=True)

    # Determine which feature column to use based on available columns.
    # We do not modify the actual DataFrame column names; we only change the labels for plotting.
    if 'average_generation' in merged_df.columns:
        x_col = 'cancer_grade'
        y_col = 'average_generation'
        feature_column = 'average_generation'
        y_label = 'Average Generation'
        scatter_title = f'Scatter Plot: {y_label} vs cancer_grade'
        density_title = f'2D Density Plot: Distribution of {y_label} across cancer_grade'
    elif 'average_distance' in merged_df.columns:
        # The actual column name in the CSV is "average_distance"
        x_col = 'cancer_grade'
        y_col = 'average_distance'
        feature_column = 'average_distance'
        file_path_lower = file_path.lower()
        if 'mean' in file_path_lower:
            y_label = 'average_mean_distance'
        elif 'best' in file_path_lower:
            y_label = 'average_best_distance'
        else:
            y_label = 'average_distance'
        scatter_title = f'Scatter Plot: {y_label} vs cancer_grade'
        density_title = f'2D Density Plot: Distribution of {y_label} across cancer_grade'
    else:
        raise ValueError("Neither 'average_distance' nor 'average_generation' found in the dataframe.")

    # Plot using the custom y-axis label and title (the DataFrame column names remain unchanged)
    scatter_plot(merged_df, x_col, y_col, 'cancer_grade', y_label_override=y_label, title_override=scatter_title, save_path='scatter_plot_cancer_grading.pdf')
    density_plot_2d(merged_df, x_col, y_col, y_label_override=y_label, title_override=density_title, save_path='density_plot_cancer_grading.pdf')


    # Run and evaluate machine learning models
    decision_tree(merged_df, feature_column)
    knn_classifier(merged_df, feature_column)
    lda_classifier(merged_df, feature_column)
    logistic_regression(merged_df, feature_column)
    random_forest(merged_df, feature_column)
    svm_classifier(merged_df, feature_column)
