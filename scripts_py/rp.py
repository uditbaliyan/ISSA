import pandas as pd
from ydata_profiling import ProfileReport
import os
import logging
import re

# Get an instance of a logger
logger = logging.getLogger(__name__)

def report(csv_file="rainfall_data_analysis/1901_1999_csv/1901uk.csv"):
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

if __name__ == "__main__":
    report(csv_file="rainfall_data_analysis/1901_1999_csv/1901uk.csv")
# end main