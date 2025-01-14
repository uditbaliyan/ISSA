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
    logger.debug("Starting report generation")
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"CSV file {csv_file.name} successfully loaded")
        
        # Disable the word cloud by using the vars parameter
        profile = ProfileReport(df, title="Profiling Report", explorative=True, vars={"text": {"wordcloud": False}})
        
        report_dir = 'templates'
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, 'pandas_profiling_report.html')
        profile.to_file(report_path)
        logger.info(f"Report successfully saved at {report_path}")
        
    except Exception as e:
        logger.error(f"Error in report generation: {str(e)}")
        raise





# # Function to remove the YData link from the HTML content
# def remove_ydata_link(html):
#     # Define the regex pattern to match the YData link section
#     pattern = r'<p class="text-body-secondary text-end">.*?<a href="https://ydata\.ai[^\"]+">YData</a>.*?</p>'
    
#     # Remove the matched pattern
#     cleaned_html = re.sub(pattern, '', html)
    
#     return cleaned_html

# # Read the HTML content from a file (replace with your file path)
# html_file_path = 'path_to_your_html_file.html'

# with open(html_file_path, 'r', encoding='utf-8') as f:
#     html_content = f.read()

# # Call the function to remove the YData link
# cleaned_html = remove_ydata_link(html_content)

# # Write the cleaned HTML back to a new file
# output_file_path = 'cleaned_html_file.html'
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     f.write(cleaned_html)

# print("YData link removed and cleaned HTML saved.")
