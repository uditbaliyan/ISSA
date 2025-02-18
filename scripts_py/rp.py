import pandas as pd
from ydata_profiling import ProfileReport
import os
import logging
import re

# Get an instance of a logger
logger = logging.getLogger(__name__)

def report(csv_file):
    """
    Purpose: Generate a profiling report from a CSV file
    """
    print(logger.debug("Starting report generation"))
    
    try:
        df = pd.read_csv(csv_file)
        # print(logger.info(f"CSV file {csv_file.name} successfully loaded"))
        
        # Disable the word cloud by using the vars parameter
        profile = ProfileReport(df, title="Profiling Report", explorative=True, vars={"text": {"wordcloud": False}})
        
        # report_dir = 'templates'
        # os.makedirs(report_dir, exist_ok=True)
        report_path = "/home/udit/Documents/Github/ISSA/ydata_Profile_Report/pandas_profiling_report.html"
        profile.to_file(report_path)
        print(logger.info(f"Report successfully saved at {report_path}"))
        
    except Exception as e:
        print(logger.error(f"Error in report generation: {str(e)}"))
        raise


def time_series(csv_file):
    """
    Purpose: 
    """

    file_path = csv_file  # Update this with your file path
    df = pd.read_csv(file_path)

    # # Convert wide format to long format
    # df_melted = df.melt(id_vars=["Days"], var_name="Year", value_name="Value")

    # # Convert Year to datetime
    # df_melted["Date"] = pd.to_datetime(df_melted["Year"], format="%Y") + pd.to_timedelta(df_melted["Days"] - 1, unit="D")

    # # Drop unnecessary columns
    # df_melted = df_melted.drop(columns=["Days", "Year"])

    # Generate profile report
    profile = ProfileReport(df, tsmode=True, sortby="Days", title="Time-Series EDA")

    # Save the report
    profile.to_file("time_series_report.html")

    print("Time-Series Analysis Report Generated Successfully!")





if __name__ == "__main__":
    path="/home/udit/Documents/Github/ISSA/data/refine_data/fid_0_data_1901_1999.csv"
    # report(csv_file=path)
    time_series(csv_file=path)

# end main