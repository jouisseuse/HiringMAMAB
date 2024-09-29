import sys
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

# Function to read and process files from a directory
def process_directory(directory):
    experiments = []
    for filename in os.listdir(directory):
        if filename.endswith('.log'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = file.readlines()
            entropy_values = [0] # 0 for round 0
            for line in data:
                match = re.search(r'offers entropy all:\s*([\d\.]+)', line)
                if match:
                    entropy_values.append(float(match.group(1)))
            experiments.append(entropy_values)
    return np.array(experiments)

# Get directories from command line arguments
directory1 = sys.argv[1]  # First input directory (e.g., decentralized)
directory2 = sys.argv[2]  # Second input directory (e.g., distributed)

# Process both directories
all_experiments_1 = process_directory(directory1)
all_experiments_2 = process_directory(directory2)

# Create DataFrames for each set of experiments
df1 = pd.DataFrame(all_experiments_1.T, columns=[f'Simulation_{i+1}' for i in range(all_experiments_1.shape[0])])
df2 = pd.DataFrame(all_experiments_2.T, columns=[f'Simulation_{i+1}' for i in range(all_experiments_2.shape[0])])

# Save DataFrames to CSV files
csv_output1 = os.path.join(directory1, "offer_results.csv")
csv_output2 = os.path.join(directory2, "offer_results.csv")
df1.to_csv(csv_output1, index=False)
df2.to_csv(csv_output2, index=False)

print(f'Data has been saved to {csv_output1} and {csv_output2}')

# Calculate means and 95% confidence intervals for both sets
def calculate_statistics(experiments):
    means = np.mean(experiments, axis=0)
    confidence_interval = 0.95
    n = len(experiments)
    se = stats.sem(experiments, axis=0)  # Standard Error
    h = se * stats.t.ppf((1 + confidence_interval) / 2., n-1)  # Confidence interval range
    return means, h

means1, h1 = calculate_statistics(all_experiments_1)
means2, h2 = calculate_statistics(all_experiments_2)

# Calculate baseline value
max_gender_entropy = stats.entropy([0.5, 0.5], base=2)
max_race_entropy = stats.entropy([1/6] * 6, base=2)
max_age_entropy = stats.entropy([1/5] * 5, base=2)
total_max_entropy = max_gender_entropy + max_race_entropy + max_age_entropy
baseline_value = total_max_entropy

# Plot data trends, confidence intervals, and baseline
plt.figure(figsize=(12, 8))
plt.plot(means1, label='Mean Entropy Values (Decentralized)')
plt.fill_between(range(len(means1)), means1 - h1, means1 + h1, color='b', alpha=0.2, label='95% Confidence Interval (Decentralized)')
plt.plot(means2, label='Mean Entropy Values (Distributed)')
plt.fill_between(range(len(means2)), means2 - h2, means2 + h2, color='g', alpha=0.2, label='95% Confidence Interval (Distributed)')
plt.axhline(y=baseline_value, color='r', linestyle='--', label=f'Baseline (Max: {baseline_value:.2f})')
plt.title('Offers Entropy Trend with 95% Confidence Interval and Baseline')
plt.xlabel('Round')
plt.ylabel('Entropy Value')
plt.legend()
plt.grid(True)

# Save the plot instead of displaying it
plot_output = os.path.join(directory1, "entropy_trend.png")
plt.savefig(plot_output)
plt.close()  # Close the plot to free memory

print(f'Plot has been saved to {plot_output}')