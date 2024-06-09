import pandas as pd

# Load the CSV files into pandas DataFrames
df1 = pd.read_csv('./data/TotalFeatures-ISCXFlowMeter.csv')
df2 = pd.read_csv('./DATASET/MALWARE-1/1.csv')

# Get the dtypes of columns in each DataFrame
dtypes_df1 = df1.dtypes
dtypes_df2 = df2.dtypes

# Compare the dtypes for each column
different_columns = []
for col in df1.columns:
    if col in df2.columns:
        if dtypes_df1[col] != dtypes_df2[col]:
            different_columns.append(col)
    else:
        print(f"Column '{col}' does not exist in file2.csv.")

if len(different_columns) > 0:
    print("The data types are different for the following columns:")
    for col in different_columns:
        print(
            f"- {col}: {dtypes_df1[col]} in file1.csv, {dtypes_df2[col]} in file2.csv")
else:
    print("The data types of columns in both CSV files are the same.")
