import matplotlib.pyplot as plt
import pandas as pd


def plot_cancer_stage_wise_bar_chart():
    """
    Plot a bar chart with cancer stage-wise distribution in a specified order.
    """
    csv_path = '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv'  # CSV with cancer types
    df_cancer_data = pd.read_csv(csv_path)

    # Define the preferred order of cancer stages
    stage_order = ["T0","Ta", "Tis", "T1", "T2", "T3", "T4"]

    # Ensure 'Final Path' is categorical with the defined order
    df_cancer_data["Final Path"] = pd.Categorical(df_cancer_data["Final Path"], categories=stage_order, ordered=True)

    # Count the number of patients per stage (in the defined order)
    stage_wise_count = df_cancer_data["Final Path"].value_counts().reindex(stage_order, fill_value=0)

    # Bar chart
    plt.bar(stage_wise_count.index, stage_wise_count.values)

    # Add labels and title
    plt.xlabel('Cancer stage',fontweight='bold')
    plt.ylabel('Number of patients',fontweight='bold')

    # Add data labels on top of each bar
    for i, count in enumerate(stage_wise_count.values):
        plt.text(i, count + 0.5, str(count), ha='center')

    # Show the plot
    plt.tight_layout()
    plt.savefig("./Results/stage_wise_patient_distribution_new.png", format="png")
    plt.show()


if __name__ == '__main__':
    plot_cancer_stage_wise_bar_chart()
