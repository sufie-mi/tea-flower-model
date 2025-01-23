import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.rc('font', family='Times New Roman')
def calculate_proportion(input_file_path):
    df = pd.read_csv(input_file_path)

    # Calculate the number of samples where labels column and predict_stages column are equal
    correct_predictions = (df['labels'] == df['predict_stages+1']).sum()

    # Calculate the total number of samples
    total_samples = len(df)

    # Calculate the proportion
    proportion = correct_predictions / total_samples

    return proportion

# Draw confusion matrix
df = pd.read_csv(r"flower_stage.csv")

# Extract predicted classes and actual classes
predictions = df['predict_stages+1']
actuals = df['labels']

# Calculate confusion_matrix
cm = confusion_matrix(actuals, predictions, labels=[1, 2, 3, 4, 5])

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.xlabel('Predicted Label',  size=14)
plt.ylabel('Original Label', size=14)
plt.title('Confusion Matrix', size=16)

# save and show confusion_matrix
folder_path = r"save_path"  
file_name = "confusion_matrix.svg" 
file_path = f"{folder_path}\\{file_name}"

plt.savefig(file_path, bbox_inches='tight')
plt.show()

# evaluate
input_file_path = r"flowerstage.csv"  # Replace with your CSV file path
proportion = calculate_proportion(input_file_path)
print(f'Proportion of samples where Labels and Predict Stages are equal : {proportion:.2%}')