import numpy as np
from matplotlib import pyplot as plt
from classification_methods.classification import perform_classification

accuracy_dict, f1_score_dict = perform_classification()
classifiers = list(accuracy_dict["cancer_invasion"].keys())

# Define bar width
bar_width = 0.30

# Set positions for the bars
index = np.arange(len(classifiers))

# Define tasks
tasks = ["cancer_invasion", "cancer_non_cancerous", "cancer_stage"]

# Loop through each task and create the plots
for task in tasks:
    plt.figure(figsize=(10, 5))
    plt.bar(
        index - bar_width / 2, [accuracy_dict[task][classifier] for classifier in classifiers], bar_width,
        label='Accuracy'
        )
    plt.bar(
        index + bar_width / 2, [f1_score_dict[task][classifier] for classifier in classifiers], bar_width,
        label='F1 Score'
        )

    # Add title, labels, and legend
    plt.title(f"{task.replace('_', ' ').title()}")
    plt.xlabel("Classifiers")
    plt.ylabel("Scores")
    plt.xticks(index, classifiers)
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Show the plot
    plt.show()
