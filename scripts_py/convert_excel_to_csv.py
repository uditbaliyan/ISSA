import os
import pandas as pd

def convert_excel_to_csv(folder_path):
    """
    Converts all Excel files in the specified folder to CSV format.
    
    Parameters:
        folder_path (str): Path to the folder containing Excel files.
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    
    # Get a list of all Excel files in the folder
    excel_files = [file for file in os.listdir(folder_path) if file.endswith(('.xlsx', '.xls'))]
    
    if not excel_files:
        print("No Excel files found in the folder.")
        return
    
    # Create an output folder for CSV files
    output_folder = "/home/udit/Documents/Github/ISSA/rainfall_data_analysis/1901_1999_csv"
    os.makedirs(output_folder, exist_ok=True)
    
    for excel_file in excel_files:
        try:
            # Full path to the Excel file
            excel_path = os.path.join(folder_path, excel_file)
            
            # Read the Excel file
            data = pd.read_excel(excel_path)
            
            # Generate a CSV file name
            csv_file_name = os.path.splitext(excel_file)[0] + ".csv"
            csv_path = os.path.join(output_folder, csv_file_name)
            
            # Save as CSV
            data.to_csv(csv_path, index=False)
            print(f"Converted '{excel_file}' to '{csv_file_name}'")
        except Exception as e:
            print(f"Failed to convert '{excel_file}': {e}")
    
    print(f"All files have been converted. CSV files are saved in '{output_folder}'.")

# Usage example
folder_path ="/home/udit/Downloads/rainfall/rerainfall_data"
# folder_path = input("Enter the folder path containing Excel files: ").strip()
convert_excel_to_csv(folder_path)
