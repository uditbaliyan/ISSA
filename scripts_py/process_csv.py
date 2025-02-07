import os
import pandas as pd

def process_csv(file_path):
    """
    Reads a CSV file, removes the first row and rows after the 86th, and rewrites the CSV.

    Parameters:
        file_path (str): Path to the CSV file.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path,header=1)
        
        # Remove the first row and rows after the 86th (keeping rows 1-86)
        # modified_data = data.iloc[0:84]  # Rows are indexed starting from 0
        modified_data = data
        # Rewrite the CSV with the modified data
        modified_data.to_csv(file_path, index=False)
        
        print(f"File '{file_path}' has been processed successfully.")
    except Exception as e:
        print(f"An error occurred while processing the file '{file_path}': {e}")

def process_all_csv_in_folder(folder_path):
    """
    Processes all CSV files in a given folder.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return
    
    # List all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in the folder '{folder_path}'.")
        return
    
    # Process each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        process_csv(file_path)

# Usage example
folder_path ="/home/udit/Documents/Github/ISSA/rainfall_data_analysis/1901_1999_csv"
# folder_path = input("Enter the folder path containing CSV files: ").strip()
process_all_csv_in_folder(folder_path)
print(f"Done")
