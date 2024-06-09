import pandas as pd

# Define the file paths for your CSV files
file1_path = './data/TotalFeatures-ISCXFlowMeter.csv'
file2_path = 'output_file.csv'

# Load the CSV files into Pandas DataFrames
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Get the list of features (column headers) for each DataFrame
features_df1 = list(df1.columns)
features_df2 = list(df2.columns)

# Compare the two lists and find indices where they do not match
mismatch_indices = [i for i, (feat1, feat2) in enumerate(
    zip(features_df1, features_df2)) if feat1 != feat2]

# Print the list of features for each file
print("Features in first file:")
print(features_df1)
print("\nFeatures in second file:")
print(features_df2)

if len(mismatch_indices) > 0:
    print("\nIndices where features do not match:")
    print(mismatch_indices)
else:
    print("\nAll features match between the two files.")
