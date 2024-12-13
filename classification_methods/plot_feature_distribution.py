import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from classification_methods.features_for_classification import max_no_of_rois, selected_feature


def plot_single_feature_distribution(cancer_df, feature):
    """
    Plots the distribution of a single selected feature across cancer stages

    """
    # Plotting the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='cancer_type', y=feature, data=cancer_df, hue='cancer_type', width=.4, showmeans=True,
        meanprops={"marker": "+",
                   "markeredgecolor": "black",
                   "markersize": "5"}
        )

    # Adding labels and title
    plt.xlabel("Cancer Stage", fontsize=12)
    plt.ylabel(f"{feature.capitalize()}  Values", fontsize=12)
    plt.title(f"Distribution of {feature.capitalize()}  Across Cancer Stages", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.savefig(f"./Results/{feature}_feature_distribution.pdf", format="pdf")
    # Show the plot
    plt.show()


def plot_full_feature_distribution(cancer_df):
    """
    Plots the distribution of all the features from the selected feature across cancer stages

    """
    # Melt the DataFrame to long-form format for seaborn boxplot
    melted_df = cancer_df.melt(
        id_vars=["cancer_type"],
        value_vars=[f"{selected_feature}_{i}" for i in range(1, 21)],
        var_name="Feature",
        value_name="Value"
        )

    # Plotting the box plot
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=melted_df, x="Feature", y="Value", hue="cancer_type", palette="Set2")

    # Adding labels and title
    plt.xlabel("Features", fontsize=12)
    plt.ylabel(f"{selected_feature.title()}  Values", fontsize=12)
    plt.title(f"Distribution of {selected_feature.capitalize()} Features Across Cancer Stages", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    #plt.savefig(f"./Results/{selected_feature}_feature_distribution.pdf", format="pdf")
    plt.show()


def plot_cancer_normal_feature_distribution(feature):
    full_data = pd.read_csv(f"./features/glcm_all_{selected_feature}_features_{max_no_of_rois}_rois.csv")

    # Plotting the box plot for each stage with separate cancer and healthy groups
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        x='cancer_type', y=feature, data=full_data, hue='label', width=0.6, showmeans=True,
        meanprops={"marker": "+", "markeredgecolor": "black", "markersize": "8"}
        )

    # Adding labels and title
    plt.xlabel("Cancer Stage", fontsize=12)
    plt.ylabel(f"{selected_feature.capitalize()} Values", fontsize=12)
    plt.title(f"Distribution of {feature.capitalize()} Across Cancer Stages", fontsize=14)

    # Dynamically retrieve the handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels()

    # Map 0 to "Normal" and 1 to "Cancer" for the legend
    new_labels = ['Control' if label == '0' else 'Lesion' for label in labels]

    # Apply the updated legend
    plt.legend(handles, new_labels, title='Label', loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save and show the plot
    #plt.savefig(f"./Results/{feature}_feature_distribution_control_vs_lesion.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    # Read the CSV file
    cancer_df = pd.read_csv(
        f"./features/glcm_cancer_{selected_feature}_features_{max_no_of_rois}_rois_with_sub_types.csv"
        )

    # Dynamically select the feature column
    feature = f"{selected_feature}_13"

    plot_full_feature_distribution(cancer_df)
    plot_single_feature_distribution(cancer_df, feature)
    plot_cancer_normal_feature_distribution(feature)
