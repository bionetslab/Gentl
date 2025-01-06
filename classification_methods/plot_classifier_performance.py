import numpy as np
from matplotlib import pyplot as plt
from classification_methods.classification import perform_classification


def plot_all_classifier_performance(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag=False):
    """
    Plot accuracy and F1 score for each classifier for 5 different tasks with value labels on bars
    """
    # Perform classification to retrieve accuracy and F1 score dictionaries
    figure_names_list = {"average_best_absolute_distance_results": "Gentl Best Distance",
                         "average_generation_absolute_distance_results": "Gentl Average Generation",
                         "average_mean_absolute_distance_results": "Gentl Mean Distance"}
    accuracy_dict, f1_score_dict = perform_classification(
        selected_feature, max_no_of_rois, gentl_result_param, gentl_flag
        )
    classifiers = list(accuracy_dict["cancer_invasion"].keys())

    # Define bar width and positions
    bar_width = 0.35
    index = np.arange(len(classifiers))

    # Define tasks
    tasks = ["cancer_invasion", "cancer_vs_non_cancerous", "cancer_stage",
             "cancer_early_vs_late_stage", "ptc_vs_mibc"]
    if gentl_flag:  # if gentl_flag is True, remove cancer_vs_non_cancerous
        tasks = ["cancer_invasion", "cancer_stage",
                 "cancer_early_vs_late_stage", "ptc_vs_mibc"]
    # Create a figure with a grid layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 16)) if gentl_flag else plt.subplots(3, 2, figsize=(20, 20))

    figure_name = figure_names_list[gentl_result_param] if gentl_flag else "GLCM Features"
    fig.suptitle(
        f"Classifier Performance using {figure_name} | Selected Feature: {selected_feature.title()} | Max ROIs: {max_no_of_rois}",
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
    fig.delaxes(axes[5]) if not gentl_flag else None

    # Add a single legend below the title but above the subplots
    handles = [plt.Rectangle((0, 0), 1, 1, color="tab:blue", label="Accuracy"),
               plt.Rectangle((0, 0), 1, 1, color="tab:orange", label="F1 Score")]
    fig.legend(handles=handles, loc="upper center", fontsize=12, ncol=2, bbox_to_anchor=(0.5, 0.92))

    # Adjust layout to ensure no overlap
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Reserve space for the title and legend
    if gentl_flag:
        plt.savefig(
            f"./Results/{max_no_of_rois}/gentl_results/{gentl_result_param}_classifier_performance_{selected_feature}_{max_no_of_rois}.pdf",
            format="pdf"
            )
    else:
        plt.savefig(
            f"./Results/{max_no_of_rois}/classifier_performance_{selected_feature}_{max_no_of_rois}.pdf", format="pdf"
            )
    # plt.show()


if __name__ == "__main__":
    gentl_result_param_list = ["average_best_absolute_distance_results", "average_generation_absolute_distance_results",
                               "average_mean_absolute_distance_results"]
    glcm_features_list = ["dissimilarity", "correlation", "energy", "homogeneity", "contrast"]
    roi_list = [10, 20, 30, 40, 50]
    """Set the flag as True if performing classification on gentl results, set to false if performing classification on glcm"""
    gentl_flag = True  #
    plot_all_classifier_performance(
        selected_feature=glcm_features_list[1], max_no_of_rois=roi_list[4],
        gentl_result_param=gentl_result_param_list[2], gentl_flag=gentl_flag
        )
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=20)
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=30)
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=40)
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=50)
    # [dissimilarity,correlation,energy,homogeneity,contrast]
    # can be set to 10,20,30,40,50
