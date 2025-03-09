import os
import numpy as np
from matplotlib import pyplot as plt
from classification_methods.classification import perform_classification


def plot_performance_across_rois_per_task(selected_feature):
    """
        Plot and print the F1 score using GLCM and gentl results for each ROI count for the selected glcm feature for all tasks.
    Args:
        selected_feature: glcm feature used for classification and as input for GA
    """
    gentl_algo_result_param_dict = {"Best Distance": "average_best_absolute_distance_results",
                                    "Mean Distance": "average_mean_absolute_distance_results",
                                    "Max Generations": "average_generation_absolute_distance_results"}
    roi_values = [10, 20, 30, 40, 50]
    f1_score_all_param_dict = {}
    tasks = ["cancer_invasion", "cancer_stage", "cancer_early_vs_late_stage", "ptc_vs_mibc"]
    classifiers = None

    for max_no_of_rois in roi_values:
        _, f1_score_glcm_features_dict = perform_classification(
            selected_feature, max_no_of_rois, None, gentl_flag=False
            )
        if classifiers is None:
            classifiers = list(f1_score_glcm_features_dict["cancer_invasion"].keys())

        f1_score_all_param_dict[max_no_of_rois] = {"GLCM Feature": f1_score_glcm_features_dict}

        for key, gentl_result_param in gentl_algo_result_param_dict.items():
            _, f1_score_gentl_dict = perform_classification(
                selected_feature, max_no_of_rois, gentl_result_param, gentl_flag=True
                )
            f1_score_all_param_dict[max_no_of_rois][key] = f1_score_gentl_dict
    classifier_name_map = {
        "logistic_regression": "LR",
        "svm_classifier": "SVM",
        "knn_classifier": "KNN",
        "decision_tree": "DT",
        "random_forest": "RF",
        "lda_classifier": "LDA"
        }

    # Convert classifier list to readable names
    formatted_classifiers = [classifier_name_map.get(c, c) for c in classifiers]
    group_spacing = 0.25

    for task_name in tasks:
        fig, axes = plt.subplots(len(roi_values), 1, figsize=(10, 14))

        fig.suptitle(
            f"Classifier Performance | Selected Feature: {selected_feature.title()}",
            fontsize=16, fontweight='bold', y=0.98
            )

        axes = axes.flatten()
        for i, max_no_of_rois in enumerate(roi_values):
            classifier_wise_f1_values_dict = {}
            ax = axes[i]

            for key, f1_score_dict in f1_score_all_param_dict[max_no_of_rois].items():
                classifier_wise_f1_values_dict[key] = [f1_score_dict[task_name][classifier] for classifier in
                                                       classifiers]

            x = np.arange(len(classifiers))
            width = 0.18
            multiplier = 0

            for label, values in classifier_wise_f1_values_dict.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, values, width, label=label)
                formatted_labels = [f"{v:.2f}" if v % 1 != 0 else f"{v:.1f}" for v in values]
                ax.bar_label(rects, labels=formatted_labels, fontsize=10, padding=2.5, rotation=90)
                multiplier += 1

            ax.set_title(f"ROI {max_no_of_rois}", fontsize=12, fontweight='bold')
            ax.set_xticks(x + (width * (multiplier - 1) / 2), formatted_classifiers, fontsize=12)
            if task_name == "cancer_stage":
                ax.set_ylim(0, 40)
            else:
                ax.set_ylim(0, 100)
            ax.set_ylabel("F1-scores (%)", fontsize=13, fontweight='bold')
            x = x + group_spacing

        # Add a single legend below the title but above the subplots
        handles = [plt.Rectangle((0, 0), 0.5, 0.5, color="tab:blue", label=selected_feature.title()),
                   plt.Rectangle((0, 0), 0.5, 0.5, color="tab:orange", label="Average best distance"),
                   plt.Rectangle((0, 0), 0.5, 0.5, color="tab:green", label="Average mean distance"),
                   plt.Rectangle((0, 0), 0.5, 0.5, color="tab:red", label="Average generation")
                   ]
        # fig.legend(handles=handles, loc="upper center", fontsize=12, ncol=2, bbox_to_anchor=(0.5, 1))
        fig.legend(
            handles=handles, loc="upper center", fontsize=12, ncol=2, bbox_to_anchor=(0.5, 0.95),
            handleheight=0.8, handlelength=1, borderpad=0.25
            )

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        # plt.savefig(
        #     f"./report_results/{task_name}/report_classifier_performance_{selected_feature}_across_rois.pdf", format="pdf"
        #     )
        plt.show()


if __name__ == "__main__":
    glcm_features_list = ["dissimilarity", "correlation", "energy", "homogeneity", "contrast"]
    roi_list = [10, 20, 30, 40, 50]
    for i in range(0, 5):
        plot_performance_across_rois_per_task(
            selected_feature=glcm_features_list[i]
            )
