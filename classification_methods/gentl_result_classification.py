import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classification_methods.features_for_classification import selected_feature


def classify_cancer_stage(Dataframe_cancer_with_distance, metric):
    # -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------

    models = {
        "svm_classifier": SVC(
            decision_function_shape='ovo', kernel="rbf", C=1, gamma=0.1, random_state=42, class_weight='balanced'
            ),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", solver="liblinear", multi_class="ovr", random_state=42
            ),
        "knn_classifier": KNeighborsClassifier(n_neighbors=13),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=42, max_depth=2, min_samples_leaf=15
            ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", random_state=42, max_depth=3, min_samples_leaf=5, n_estimators=25
            ),
        "lda_classifier": LinearDiscriminantAnalysis(shrinkage=0.5, solver='lsqr')
        }

    X = Dataframe_cancer_with_distance[[metric]]  # no need to drop index
    y = Dataframe_cancer_with_distance["cancer_type"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    accuracy_dict = {}
    f1_score_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate accuracy and F1-score
        accuracy = accuracy_score(y_test, y_pred) * 100
        f1_score_ = f1_score(y_test, y_pred, average="weighted") * 100  # specify average for multiclass problems

        accuracy_dict[name] = round(accuracy, 2)
        f1_score_dict[name] = round(f1_score_, 2)

    return accuracy_dict, f1_score_dict


def plot_all_classifier_performance(accuracy_dict, f1_score_dict,selected_feature):
    """
    Plot accuracy and F1 score for each classifier for 3 different tasks
    """
    classifiers = list(accuracy_dict.keys())

    # Define bar width
    bar_width = 0.30

    # Set positions for the bars
    index = np.arange(len(classifiers))


    plt.figure(figsize=(10, 5))
    plt.bar(
            index - bar_width / 2, [accuracy_dict[classifier] for classifier in classifiers], bar_width,
            label='Accuracy'
            )
    plt.bar(
            index + bar_width / 2, [f1_score_dict[classifier] for classifier in classifiers], bar_width,
            label='F1 Score'
            )

    # Add title, labels, and legend
    plt.title(f"Gentl result using {selected_feature}")
    plt.xlabel("Classifiers")
    plt.ylabel("Scores")
    plt.xticks(index, classifiers)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"./Results/gentl_result_{selected_feature}_classifier_performance.pdf", format="pdf")
    # Show the plot
    plt.show()


if __name__ == "__main__":
    metric = "average_generation"  # average_generation,average_distance
    selected_feature = "correlation" #[dissimilarity,correlation,energy,contrast,homogeneity]
    Dataframe_cancer_with_distance = pd.read_csv(
        f"../glcm_average_gentl_results/{selected_feature}_{metric}_results.csv"
        )
    csv_path = '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv'  # csv with cancer types
    df_cancer_types = pd.read_csv(csv_path)

    Dataframe_cancer_with_distance = pd.merge(
        Dataframe_cancer_with_distance, df_cancer_types[["Final Path", "Anonymized ID"]], left_on='patient_id',
        right_on="Anonymized ID", how='left'
        )
    Dataframe_cancer_with_distance = Dataframe_cancer_with_distance.rename(
        columns={"Final Path": "cancer_type"}
        )
    Dataframe_cancer_with_distance = Dataframe_cancer_with_distance.drop("Anonymized ID", axis=1)
    Dataframe_cancer_with_distance = Dataframe_cancer_with_distance.set_index("patient_id")

    accuracy_dict, f1_score_dict = classify_cancer_stage(Dataframe_cancer_with_distance, metric)
    plot_all_classifier_performance(accuracy_dict, f1_score_dict,selected_feature)
