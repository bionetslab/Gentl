import os

import numpy as np
from matplotlib import pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width

from classification_methods.classification import perform_classification
import pandas as pd
import seaborn as sns


def plot_all_classifier_performance(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag=False):
    pass


def plot_performance_task_wise(selected_feature, max_no_of_rois):

    gentl_algo_result_param_dict = {"Best Distance": "average_best_absolute_distance_results",
                                    "Mean Distance": "average_mean_absolute_distance_results",
                                    "Max Generations": "average_generation_absolute_distance_results"}
    f1_score_all_param_dict = {}
    _, f1_score_glcm_features_dict = perform_classification(
        selected_feature, max_no_of_rois, None, gentl_flag=False
        )
    f1_score_all_param_dict["GLCM Feature"] = f1_score_glcm_features_dict
    for key, gentl_result_param in gentl_algo_result_param_dict.items():
        _, f1_score_gentl_dict = perform_classification(
            selected_feature, max_no_of_rois, gentl_result_param, gentl_flag=True
            )
        f1_score_all_param_dict[key] = f1_score_gentl_dict
    classifiers = list(f1_score_glcm_features_dict["cancer_invasion"].keys())

    group_spacing = 0.3  # Add extra space between bar groups for separate tasks

    # Define tasks
    tasks = ["ptc_vs_mibc"]
    # Create a figure with a grid layout
    fig, axes = plt.subplots(1, 1, figsize=(9, 4))

    # fig.suptitle(
    #     f"Classifier Performance | Selected Feature: {selected_feature.title()} | Max ROIs: {max_no_of_rois}",
    #     fontsize=16, fontweight='bold', y=0.98
    #     )  # Main title for the plot

    # Flatten axes for easier indexing = axes has the shape as (3,2)
    # axes = axes.flatten()

    # Loop through tasks and plot in their respective subplots
    for i, task in enumerate(tasks):
        classifier_wise_f1_values_dict = {}
        ax = axes
        for key, f1_score_dict in f1_score_all_param_dict.items():
            classifier_wise_f1_values_dict[key] = [f1_score_dict[task][classifier] for classifier in classifiers]

        x = np.arange(len(classifiers))  # the label locations
        width = 0.20  # the width of the bars, smaller than before
        multiplier = 0

        for label, values in classifier_wise_f1_values_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, values, width, label=label)
            ax.bar_label(rects, fontsize=8, padding=3, rotation=90)
            multiplier += 1

        # Add task-specific title, labels, and ticks
        # ax.set_title(task.replace("_", " ").title(), fontsize=12, fontweight='bold')
        ax.set_xticks(x + (width * (multiplier - 1) / 2), classifiers)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        # if i % 2 == 0:  # Add y-axis label only to the first column
        ax.set_ylabel("F1 Scores (%)",fontsize=10, fontweight='bold')
        ax.set_xlabel("Classifiers",fontsize=10, fontweight='bold',labelpad=10)
        # Adjust the position of each bar group to include the extra spacing
        x = x + group_spacing

    # Add a single legend below the title but above the subplots
    handles = [plt.Rectangle((0, 0), 1, 1, color="tab:blue", label="GLCM Feature"),
               plt.Rectangle((0, 0), 1, 1, color="tab:orange", label="Best Distance"),
               plt.Rectangle((0, 0), 1, 1, color="tab:green", label="Mean Distance"),
               plt.Rectangle((0, 0), 1, 1, color="tab:red", label="Max Generations")
               ]
    fig.legend(handles=handles, loc="upper center", fontsize=10, ncol=4, bbox_to_anchor=(0.5, 0.98))

    # Adjust layout to ensure no overlap
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Reserve space for the title and legend

    plt.savefig(
        f"./Results/{tasks[0]}.pdf",
        format="pdf"
        )

    # plt.show()


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


def plot_performance_across_all_glcm_features(max_no_of_rois, gentl_result_param):
    """
    Plot F1 scores across all GLCM features for each classifier using a selected GenTL result.

    :param max_no_of_rois: Number of ROIs per image
    :param gentl_algo_result_param: Selected GenTL result parameter
    """
    # Define all GLCM features (you need to specify them)
    glcm_features = ["dissimilarity", "correlation", "energy", "homogeneity", "contrast"]  # Add more if needed

    # Store F1 scores for all features
    f1_score_all_features_dict = {}

    # Loop over each GLCM feature
    for feature in glcm_features:
        _, f1_score_glcm_dict = perform_classification(
            feature, max_no_of_rois, gentl_result_param, gentl_flag=True
            )
        f1_score_all_features_dict[feature] = f1_score_glcm_dict

    classifiers = list(f1_score_all_features_dict[glcm_features[0]]["cancer_invasion"].keys())

    # Define tasks
    tasks = ["cancer_invasion", "cancer_stage", "cancer_early_vs_late_stage", "ptc_vs_mibc"]

    # Create a figure with a grid layout
    fig, axes = plt.subplots(len(tasks), 1, figsize=(12, 18))
    fig.suptitle(
        f"Classifier Performance Across All GLCM Features for {gentl_result_param} \n Max ROIs: {max_no_of_rois}",
        fontsize=16, fontweight='bold', y=0.98
        )  # Main title for the plot

    # Flatten axes for easier indexing
    axes = np.ravel(axes)

    width = 0.15  # Width of bars
    colors = plt.cm.tab10.colors  # Use Matplotlib colormap

    # Loop through tasks and plot in their respective subplots
    for i, task in enumerate(tasks):
        ax = axes[i]
        feature_wise_f1_values_dict = {}

        for feature in glcm_features:
            feature_wise_f1_values_dict[feature] = [
                f1_score_all_features_dict[feature][task][classifier] for classifier in classifiers
                ]

        x = np.arange(len(classifiers))  # X-axis locations
        multiplier = 0

        for feature, values in feature_wise_f1_values_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, values, width, label=feature, color=colors[multiplier % len(colors)])
            ax.bar_label(rects, fontsize=8, padding=3, rotation=90, fmt="%.2f")
            multiplier += 1

        # Add task-specific title and labels
        ax.set_title(task.replace("_", " ").title(), fontsize=12, fontweight='bold')
        ax.set_xticks(x + (width * (multiplier - 1) / 2), classifiers)
        ax.set_ylabel("F1 Score %")
        ax.set_ylim(0, max(1.0, ax.get_ylim()[1] * 1.15))  # Prevent scaling issues

    # Add a single legend below the title but above the subplots
    fig.legend(
        labels=glcm_features,
        loc="upper center",
        fontsize=12,
        ncol=5,
        bbox_to_anchor=(0.5, 0.92)
        )

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Ensure the save directory exists
    save_path = f"./Results/{max_no_of_rois}/glcm_features/"
    os.makedirs(save_path, exist_ok=True)

    # Save the plot
    plt.savefig(
        f"{save_path}classifier_performance_across_glcm_features_{max_no_of_rois}_{gentl_result_param}.pdf",
        format="pdf"
        )

    # plt.show()


if __name__ == "__main__":
    gentl_result_param_list = ["average_best_absolute_distance_results", "average_generation_absolute_distance_results",
                               "average_mean_absolute_distance_results"]
    glcm_features_list = ["dissimilarity", "correlation", "energy", "homogeneity", "contrast"]
    roi_list = [10, 20, 30, 40, 50]
    """Set the flag as True if performing classification on gentl results, set to false if performing classification on glcm"""
    gentl_flag = True  #
    only_f1_score_all_task = True # to plot f1 score across tasks
    only_f1_score_roi_set = False # a single plot for 10,20,30,40 and 50 rois
    for i in range(0,5):
        plot_performance_across_all_glcm_features(max_no_of_rois=roi_list[i], gentl_result_param=gentl_result_param_list[2])
    # if only_f1_score_roi_set:
    #     plot_performance_across_all_rois(selected_feature=glcm_features_list[4])
    # elif only_f1_score_all_task:  # to plot f1 score across all the task in 1 plot
    #     plot_performance_task_wise(
    #         selected_feature=glcm_features_list[3], max_no_of_rois=roi_list[1]
    #         )
    # else:  # to plot performance accuracy and f1 score task wise
    #     plot_all_classifier_performance(
    #         selected_feature=glcm_features_list[1], max_no_of_rois=roi_list[0],
    #         gentl_result_param=gentl_result_param_list[2], gentl_flag=gentl_flag
    #         )
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=20)
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=30)
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=40)
    # plot_all_classifier_performance(selected_feature="contrast", max_no_of_rois=50)
    # [dissimilarity,correlation,energy,homogeneity,contrast]
    # can be set to 10,20,30,40,50
