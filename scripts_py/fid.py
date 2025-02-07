import pandas as pd
import os

# Directory containing the CSV files
directory = 'data/rainfall_data_analysis_csv/1901_1999_csv'

# Initialize a dictionary to store FID data
fid_data = {}

for fid in range(0, 83):
    # comment: 

    # Loop through each year from 1901 to 1999
    for year in range(1901, 2000):
        filename = f'{year}uk.csv'
        filepath = os.path.join(directory, filename)
        
        # Check if the file exists
        if os.path.exists(filepath):
            # Read the CSV file and drop unnamed columns
            df = pd.read_csv(filepath).dropna(axis=1, how='all')
            
            # Ensure the 'FID' column exists
            if 'FID' in df.columns:
                fid_data[year] = df.loc[fid]
            else:
                print(f"Warning: 'FID' column not found in {filename}")

    # Convert dictionary to DataFrame
    fid_df = pd.DataFrame(fid_data)

    # Save the DataFrame to a new CSV file
    output_filename = f'data/refine_data/fid_{fid}_data_1901_1999.csv'
    fid_df.to_csv(output_filename, index=False)

    print(f"FID data has been saved to {output_filename}")
    # end for

# df = pd.read_csv("data/rainfall_data_analysis_csv/1901_1999_csv/1901uk.csv")
# print(f"{df.loc[3]}")