import os
import pandas as pd


def merge_csv_files(parent_dir, output_file):
    # Create an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()

    # Iterate through all files and directories in the parent directory
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                # Read each CSV file and append it to the merged DataFrame
                df = pd.read_csv(file_path)
                merged_df = pd.concat([merged_df, df])

    # Write the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)


# Example usage
parent_directory = './mer'
output_csv = 'dataset.csv'
merge_csv_files(parent_directory, output_csv)
