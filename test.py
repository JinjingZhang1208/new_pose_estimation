import pandas as pd

# Load the dataset
data = pd.read_csv('keypoints_output/wave.csv')

# Drop rows with NaN in 'label' column
data = data.dropna(subset=['label'])

# Check unique labels after dropping NaNs
print("Unique labels after removing NaNs:", data['label'].unique())

# Further processing or saving the cleaned data
data.to_csv('keypoints_output/wave_cleaned.csv', index=False)

