from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from classification_methods.features_for_classification import get_features_by_type, get_all_features, \
    get_features_by_sub_type


def classify_cancer_invasion():
    # #-------------------NMIBC Vs MIBC----------------------
    Dataframe_cancer_with_types = get_features_by_type()

    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_type", "cancer_type_label"]
        )  # no need to drop index
    # X = Dataframe_cancer_with_types.iloc[:, 6:8]
    y = Dataframe_cancer_with_types["cancer_type_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create KNN classifier
    model = KNeighborsClassifier(n_neighbors=9)

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_score_ = f1_score(y_test, y_pred) * 100

    # print(f"Accuracy for MIBC vs NMIBC: {accuracy:.2f}%")
    # print(f"F1-score for MIBC vs NMIBC: {f1_score_:.2f}%")
    #print(classification_report(y_test, y_pred))
    # """Accuracy for MIBC vs NMIBC: 76.92%
    # F1-score for MIBC vs NMIBC: 82.35%"""
    return accuracy, f1_score_


def classify_cancer_vs_non_cancerous():
    # #-------------------Cancer Vs Non-cancer-----------------------------------------
    full_features_dataframe = get_all_features()
    X = full_features_dataframe.drop(columns=["label","cancer_type"])  # no need to drop index
    y = full_features_dataframe["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create KNN classifier
    model = KNeighborsClassifier(n_neighbors=9)

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_score_ = f1_score(y_test, y_pred) * 100

    # print(f"Accuracy for cancer vs normal ROI: {accuracy:.2f}%")
    # print(f"F1-score for cancer vs normal ROI: {f1_score_:.2f}%")
    #print(classification_report(y_test, y_pred))
    # """Accuracy for cancer vs normal ROI: 80.00%
    # F1-score for cancer vs normal ROI: 82.61%"""
    return accuracy, f1_score_


def classify_cancer_stage():
    # -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
    Dataframe_cancer_with_types = get_features_by_sub_type()
    X = Dataframe_cancer_with_types.drop(
        columns=["label", "cancer_type", "cancer_sub_type_label"]
        )  # no need to drop index
    y = Dataframe_cancer_with_types["cancer_sub_type_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the Knn classifier
    model = KNeighborsClassifier(n_neighbors=13)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    #
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1_score_ = f1_score(y_test, y_pred, average="weighted") * 100  # specify average for multiclass problems

    # print(f"Accuracy for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: {accuracy:.2f}%")
    # print(f"F1-score for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: {f1_score_:.2f}%")
    # #print(classification_report(y_test, y_pred))
    # """Accuracy for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: 40.00%
    # F1-score for T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4: 31.67%"""
    return accuracy, f1_score_
