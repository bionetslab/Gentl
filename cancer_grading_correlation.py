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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils import class_weight

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

def decision_tree(merged_df, feature_column):
    stage_mapping = {
        "Ta": 0,  "Tis": 0, "T0": 0,
        "T1": 1,  "T2": 1,  "T3": 1, "T4": 1
    }

    merged_df['cancer_stage'] = merged_df['cancer_grade'].map(stage_mapping)
    if 'cancer_stage' not in merged_df.columns or merged_df['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = merged_df[[feature_column]]
    y = merged_df['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
                                   max_depth=2, min_samples_leaf=7, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_score_ = f1_score(y_test, y_pred, average='weighted') * 100
    print(f"Accuracy for MIBC vs NMIBC based on {feature_column}: {accuracy:.2f}%")
    print(f"F1-score for MIBC vs NMIBC based on {feature_column}: {f1_score_:.2f}%")
    print(classification_report(y_test, y_pred))

def knn_classifier(merged_df, feature_column, n_neighbors=5):
    stage_mapping = {
        "Ta": 0, "Tis": 0, "T0": 0,
        "T1": 1, "T2": 1, "T3": 1, "T4": 1
    }

    merged_df['cancer_stage'] = merged_df['cancer_grade'].map(stage_mapping)
    if 'cancer_stage' not in merged_df.columns or merged_df['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = merged_df[[feature_column]]
    y = merged_df['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "KNN")

def lda_classifier(merged_df, feature_column):
    stage_mapping = {
        "Ta": 0, "Tis": 0, "T0": 0,
        "T1": 1, "T2": 1, "T3": 1, "T4": 1
    }

    merged_df['cancer_stage'] = merged_df['cancer_grade'].map(stage_mapping)
    if 'cancer_stage' not in merged_df.columns or merged_df['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = merged_df[[feature_column]]
    y = merged_df['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "LDA")

def logistic_regression(merged_df, feature_column):
    stage_mapping = {
        "Ta": 0, "Tis": 0, "T0": 0,
        "T1": 1, "T2": 1, "T3": 1, "T4": 1
    }

    merged_df['cancer_stage'] = merged_df['cancer_grade'].map(stage_mapping)
    if 'cancer_stage' not in merged_df.columns or merged_df['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = merged_df[[feature_column]]
    y = merged_df['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Logistic Regression")

def random_forest(merged_df, feature_column, n_estimators=100):
    stage_mapping = {
        "Ta": 0, "Tis": 0, "T0": 0,
        "T1": 1, "T2": 1, "T3": 1, "T4": 1
    }

    merged_df['cancer_stage'] = merged_df['cancer_grade'].map(stage_mapping)
    if 'cancer_stage' not in merged_df.columns or merged_df['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = merged_df[[feature_column]]
    y = merged_df['cancer_stage']
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight_dict, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "Random Forest")

def svm_classifier(merged_df, feature_column):
    stage_mapping = {
        "Ta": 0, "Tis": 0, "T0": 0,
        "T1": 1, "T2": 1, "T3": 1, "T4": 1
    }

    merged_df['cancer_stage'] = merged_df['cancer_grade'].map(stage_mapping)
    if 'cancer_stage' not in merged_df.columns or merged_df['cancer_stage'].isna().any():
        raise ValueError("Error in mapping cancer grades to stages.")

    X = merged_df[[feature_column]]
    y = merged_df['cancer_stage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = SVC(kernel='linear', C=1.0, random_state=42)  # You can change the kernel and C parameter
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "SVM")

def evaluate_model(y_true, y_pred, model_name):
    print(f"\nEvaluating model: {model_name}")
    accuracy = accuracy_score(y_true, y_pred) * 100
    f1_score_ = f1_score(y_true, y_pred, average='weighted') * 100
    print(f"Accuracy for {model_name}: {accuracy:.2f}%")
    print(f"F1-score for {model_name}: {f1_score_:.2f}%")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    # Step 1: load files
    average_results_df = pd.read_csv('../Gentl/dissimilarity_average_mean_absolute_distance_results_20_rois.csv')
    cancer_grades_df = pd.read_csv(
        '../Gentl/data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv')
    # Step 2: rename columns, clear NaN.
    cancer_grades_df.rename(columns={'Anonymized ID': 'patient_id'}, inplace=True)
    cancer_grades_df.rename(columns={'Final Path': 'cancer_grade'}, inplace=True)
    cancer_grades_df.dropna(subset=['patient_id', 'cancer_grade'], inplace=True)
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
        feature_column = 'average_distance'
    elif 'average_generation' in merged_df.columns:
        x_col = 'cancer_grade'
        y_col = 'average_generation'
        feature_column = 'average_generation'
    else:
        raise ValueError("Neither 'average_distance' nor 'average_generation' found in the dataframe.")

    # plot
    scatter_plot(merged_df, x_col, y_col, 'cancer_grade')
    density_plot_2d(merged_df, x_col, y_col)
    # ML classification
    decision_tree(merged_df, feature_column)
    knn_classifier(merged_df, feature_column)
    lda_classifier(merged_df, feature_column)
    logistic_regression(merged_df, feature_column)
    random_forest(merged_df, feature_column)
    svm_classifier(merged_df, feature_column)


