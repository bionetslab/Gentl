import numpy as np
from matplotlib import pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

from classification_methods.classification import perform_classification
import pandas as pd
import seaborn as sns


def plot_all_classifier_performance(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag=False):
    pass


def plot_performance_across_all_task(selected_feature, max_no_of_rois):
    pass


def plot_performance_across_all_rois(selected_feature):
    """
    Plot F1 score using GLCM and gentl results for each of the tasks for multiple ROI values.
    :param selected_feature: GLCM feature
    """
    gentl_algo_result_param_dict = {
        "Best Distance": "average_best_absolute_distance_results",
        "Max Generations": "average_generation_absolute_distance_results",
        "Mean Distance": "average_mean_absolute_distance_results"
        }

    # Define ROI values
    roi_values = [10, 20, 30, 40, 50]

    # Define tasks to plot
    tasks = ["cancer_invasion", "cancer_stage", "cancer_early_vs_late_stage", "ptc_vs_mibc"]

    # Initialize a list to store results
    results = []
    classifier_list = []
    classifier_dict = {"logistic_regression": "Logistic Regression", "svm_classifier": "Support Vector Machine",
                       "knn_classifier": "K-Nearest Neighbors",
                       "decision_tree": "Decision Tree", "random_forest": "Random Forest",
                       "lda_classifier": "Linear Discriminant Analysis"}
    for max_no_of_rois in roi_values:
        # Get F1 scores for GLCM features
        _, f1_score_glcm_features_dict = perform_classification(
            selected_feature, max_no_of_rois, None, gentl_flag=False
            )

        # Add GLCM results to the results list
        for task, classifier_scores in f1_score_glcm_features_dict.items():
            if task in tasks:
                for classifier, score in classifier_scores.items():
                    results.append(
                        {
                            "Feature": "GLCM Feature",
                            "Method": classifier,
                            "Task": task.replace("_", " ").title(),
                            "ROI": f"ROI {max_no_of_rois}",
                            "Efficiency": score
                            }
                        )
                    if max_no_of_rois == 10:
                        classifier_list.append(classifier)

        # Get F1 scores for gentl algorithm results
        for key, gentl_result_param in gentl_algo_result_param_dict.items():
            _, f1_score_gentl_dict = perform_classification(
                selected_feature, max_no_of_rois, gentl_result_param, gentl_flag=True
                )

            # Add gentl results to the results list
            for task, classifier_scores in f1_score_gentl_dict.items():
                if task in tasks:
                    for classifier, score in classifier_scores.items():
                        results.append(
                            {
                                "Feature": key,
                                "Method": classifier,
                                "Task": task.replace("_", " ").title(),
                                "ROI": f"ROI {max_no_of_rois}",
                                "Efficiency": score
                                }
                            )

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)

    # Set Seaborn theme
    sns.set_theme(style="whitegrid")  # Set white background
    # sns.despine()  # Remove unnecessary spines for a cleaner look

    # Define custom colors for ROI
    roi_colors = {f"ROI {roi}": color for roi, color in zip(roi_values, sns.color_palette("bright", len(roi_values)))}

    # Define custom marker styles for each feature
    feature_markers = {
        "GLCM Feature": '^',  # Triangle
        "Best Distance": 's',  # Square
        "Max Generations": 'D',  # Diamond
        "Mean Distance": 'o'  # Circle
        }

    # Plot a separate catplot for each task
    for task in tasks:
        task_df = df[df["Task"] == task.replace("_", " ").title()]

        # Create the catplot
        g = sns.catplot(
            data=task_df, x="Feature", y="Efficiency", hue="ROI", col="Method",
            kind="point", height=3, aspect=1, palette=roi_colors, legend=False, markers=False
            )

        # Customize markers for each feature
        for ax, method in zip(g.axes.flat, task_df["Method"].unique()):
            subset = task_df[task_df['Method'] == method]
            for roi, color in roi_colors.items():
                roi_subset = subset[subset['ROI'] == roi]
                ax.plot(
                    roi_subset['Feature'], roi_subset['Efficiency'],
                    marker='', linestyle='-', color=color, linewidth=0.02
                    )
            for feature, marker in feature_markers.items():
                feature_subset = subset[subset['Feature'] == feature]
                ax.plot(
                    feature_subset['Feature'], feature_subset['Efficiency'],
                    marker=marker, linestyle='', color='brown', label=feature, markersize=4.5
                    )
            # Remove x-ticks and x-axis labels
            ax.set_xticks([])  # Remove x-ticks
            ax.set_xlabel('')  # Remove x-axis label

        # Add custom legends
        # ROI legend
        roi_handles = [plt.Line2D([0], [0], color=color, label=roi)
                       for roi, color in roi_colors.items()]
        # Feature markers legend
        feature_handles = [
            plt.Line2D([0], [0], marker=marker, color='brown', linestyle='', label=feature, markersize=6)
            for feature, marker in feature_markers.items()
            ]

        # Split legends into two rows
        roi_legend = roi_handles  # First row: ROIs
        feature_legend = feature_handles  # Second row: Features

        # Add legends as two rows
        g.fig.legend(
            roi_legend, [h.get_label() for h in roi_legend],
            loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=5, labelcolor="black"
            )
        g.fig.legend(
            feature_legend, [h.get_label() for h in feature_legend],
            loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, labelcolor="black"
            )

        # # Combine handles and labels explicitly
        # roi_legend_handles = roi_handles  # First row: ROI-related items
        # feature_legend_handles = feature_handles  # Second row: Feature-related items
        #
        # # Combine the handles and labels
        # combined_legend_handles = roi_legend_handles + feature_legend_handles
        # combined_legend_labels = (
        #         [h.get_label() for h in roi_legend_handles] +
        #         [h.get_label() for h in feature_legend_handles]
        # )
        #
        # # Add a single legend with explicit rows
        # g.fig.legend(
        #     combined_legend_handles, combined_legend_labels,
        #     loc='upper center', title="Legend", bbox_to_anchor=(0.5, 1.15), ncol=5  # Set ncol to 5 for proper spacing
        #     )

        # Additional plot adjustments
        g.despine(left=True)
        g.set_axis_labels("Feature", "F1 Score %", color="black", fontsize=10)
        g.set_titles("")
        # Move the x-axis label further down
        i = 0
        for ax in g.axes.flat:
            ax.set_xlabel(classifier_dict[classifier_list[i]], labelpad=15, color="black", fontsize=10)
            i += 1

        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(
            f"Classifier Performance for Task: {task.replace('_', ' ').title()} | Selected Feature: {selected_feature.title()}",
            fontsize=16, fontweight="bold", y=1.3, color="black"
            )
        g.fig.set_dpi(150)  # Increase DPI for sharper images
        g.fig.patch.set_facecolor('white')  # Ensure the figure has a clean white background

        # Save plot to file
        filename = f"./Results/{selected_feature}/{task.replace('_', ' ').title().replace(' ', '_').lower()}_{selected_feature.lower()}_performance.pdf"
        g.savefig(filename, format="pdf")
        print(f"Plot saved as {filename}")

        # Show plot
        # plt.show()


if __name__ == "__main__":
    gentl_result_param_list = ["average_best_absolute_distance_results", "average_generation_absolute_distance_results",
                               "average_mean_absolute_distance_results"]
    glcm_features_list = ["dissimilarity", "correlation", "energy", "homogeneity", "contrast"]
    roi_list = [10, 20, 30, 40, 50]
    """Set the flag as True if performing classification on gentl results, set to false if performing classification on glcm"""
    gentl_flag = True  #
    only_f1_score_all_task = True  # to plot f1 score across tasks
    only_f1_score_roi_set = True  # a single plot for 10,20,30,40 and 50 rois
    if only_f1_score_roi_set:
        plot_performance_across_all_rois(selected_feature=glcm_features_list[4])
    elif only_f1_score_all_task:  # to plot f1 score across all the task in 1 plot
        plot_performance_across_all_task(
            selected_feature=glcm_features_list[4], max_no_of_rois=roi_list[4]
            )
    else:  # to plot performance accuracy and f1 score task wise
        plot_all_classifier_performance(
            selected_feature=glcm_features_list[1], max_no_of_rois=roi_list[0],
            gentl_result_param=gentl_result_param_list[2], gentl_flag=gentl_flag
            )
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=20)
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=30)
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=40)
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=50)
    # [dissimilarity,correlation,energy,homogeneity,contrast]
    # can be set to 10,20,30,40,50
