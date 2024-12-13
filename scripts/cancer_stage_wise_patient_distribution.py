import matplotlib.pyplot as plt
import pandas as pd

csv_path = '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv'  # csv with cancer types
df_cancer_data = pd.read_csv(csv_path)

# Get the cancer stages
cancer_stages = list(df_cancer_data["Final Path"].dropna().unique())

# Get the sum of patient counts in each stage
stage_wise_count = []
for stage in cancer_stages:
    stage_wise_count.append(len(df_cancer_data[df_cancer_data["Final Path"] == stage]))

# Bar chart
plt.bar(cancer_stages, stage_wise_count, color='skyblue')

# Add labels and title
plt.xlabel('Cancer Stage')
plt.ylabel('Number of Patients')
plt.title('Number of Patients in Each Cancer Stage')

# Add data labels on top of each bar
for i, count in enumerate(stage_wise_count):
    plt.text(i, count + 0.5, str(count), ha='center')

# Show the plot
plt.tight_layout()
# plt.savefig("stage_wise_patient_distribution.pdf", format="pdf")
plt.show()
