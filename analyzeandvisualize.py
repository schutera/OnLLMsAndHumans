import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
csv_file = "Pick_a_random_number_between_0_and_9_t0.2.csv"
df = pd.read_csv(csv_file)

# Generate a heatmap based on the answers
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
heatmap = sns.histplot(data=df, x="Answer", kde=True, stat="probability", discrete=True)
heatmap.set_title("Distribution of " + csv_file)
heatmap.set_xlabel("Answer")
heatmap.set_ylabel("Probability")
plt.xlim(0, 9)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()
