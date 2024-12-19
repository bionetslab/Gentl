import numpy as np
from matplotlib import pyplot as plt
from classification_methods.classification import perform_classification


def plot_all_classifier_performance(selected_feature, max_no_of_rois):
    """
    Plot accuracy and F1 score for each classifier for 5 different tasks with value labels on bars
    """
    # Perform classification to retrieve accuracy and F1 score dictionaries
    accuracy_dict, f1_score_dict = perform_classification(selected_feature, max_no_of_rois)
    classifiers = list(accuracy_dict["cancer_invasion"].keys())

    # Define bar width and positions
    bar_width = 0.35
    index = np.arange(len(classifiers))

    # Define tasks
    tasks = ["cancer_invasion", "cancer_vs_non_cancerous", "cancer_stage",
             "cancer_early_vs_late_stage", "ptc_vs_mibc"]

    # Create a figure with a grid layout
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    fig.suptitle(
        f"Classifier Performance | Selected Feature: {selected_feature.title()} | Max ROIs: {max_no_of_rois}",
        fontsize=16, fontweight='bold', y=0.98
        )  # Main title for the plot

    # Flatten axes for easier indexing = axes has the shape as (3,2)
    axes = axes.flatten()

    # Loop through tasks and plot in their respective subplots
    for i, task in enumerate(tasks):
        ax = axes[i]
        accuracy_values = [accuracy_dict[task][classifier] for classifier in classifiers]
        f1_values = [f1_score_dict[task][classifier] for classifier in classifiers]

        # Plot bars for accuracy and F1 score
        bars1 = ax.bar(index - bar_width / 2, accuracy_values, bar_width, label="Accuracy")
        bars2 = ax.bar(index + bar_width / 2, f1_values, bar_width, label="F1 Score")

        # Add value labels on top of bars
        for bar in bars1:  # For accuracy
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, color='black'
                )
        for bar in bars2:  # For F1 score
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, color='black'
                )

        # Add task-specific title, labels, and ticks
        ax.set_title(task.replace("_", " ").title(), fontsize=12, fontweight='bold')
        ax.set_xticks(index)
        ax.set_xticklabels(classifiers, rotation=0)
        if i % 2 == 0:  # Add y-axis label only to the first column
            ax.set_ylabel("Scores %")

    # Remove empty subplot (since the last row has only one task)
    fig.delaxes(axes[5])

    # Add a single legend below the title but above the subplots
    handles = [plt.Rectangle((0, 0), 1, 1, color="tab:blue", label="Accuracy"),
               plt.Rectangle((0, 0), 1, 1, color="tab:orange", label="F1 Score")]
    fig.legend(handles=handles, loc="upper center", fontsize=12, ncol=2, bbox_to_anchor=(0.5, 0.92))

    # Adjust layout to ensure no overlap
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Reserve space for the title and legend
    plt.savefig(
        f"./Results/{max_no_of_rois}/classifier_performance_{selected_feature}_{max_no_of_rois}.pdf", format="pdf"
        )
    # plt.show()


if __name__ == "__main__":
    plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=10)
    plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=20)
    plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=30)
    plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=40)
    plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=50)
    # [dissimilarity,correlation,energy,homogeneity,contrast]
    # can be set to 10,20,30,40,50
