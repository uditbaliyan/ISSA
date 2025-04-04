import pandas as pd
# from ydata_profiling import ProfileReport
import os
import logging
import re

# Get an instance of a logger
logger = logging.getLogger(__name__)


def report(csv_file):
        report_dir = 'templates'
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, 'pandas_profiling_report.html')

# def report(csv_file):
#     """
#     Purpose: Generate a profiling report from a CSV file
#     """
#     logger.debug("Starting report generation")
    
#     try:
#         df = pd.read_csv(csv_file)
#         logger.info(f"CSV file {csv_file.name} successfully loaded")
        
#         # Disable the word cloud by using the vars parameter
#         profile = ProfileReport(df, title="Profiling Report", explorative=True, vars={"text": {"wordcloud": False}})
        
#         report_dir = 'templates'
#         os.makedirs(report_dir, exist_ok=True)
#         report_path = os.path.join(report_dir, 'pandas_profiling_report.html')
#         profile.to_file(report_path)
#         logger.info(f"Report successfully saved at {report_path}")
        
#     except Exception as e:
#         logger.error(f"Error in report generation: {str(e)}")
#         raise



# # import pandas as pd
# # import sweetviz as sv
# # import os
# # import logging

# # # Get an instance of a logger
# # logger = logging.getLogger(__name__)

# # def report(csv_file):
# #     """
# #     Purpose: Generate a profiling report from a CSV file using Sweetviz
# #     """
# #     logger.debug("Starting report generation")
    
# #     try:
# #         df = pd.read_csv(csv_file)
# #         logger.info(f"CSV file {csv_file.name} successfully loaded")
        
# #         # Generate the analysis report
# #         report = sv.analyze(df)
        
# #         report_dir = 'templates'
# #         os.makedirs(report_dir, exist_ok=True)
# #         report_path = os.path.join(report_dir, 'sweetviz_report.html')
        
# #         # Save the report to an HTML file
# #         report.show_html(report_path)
# #         logger.info(f"Report successfully saved at {report_path}")
        
# #     except Exception as e:
# #         logger.error(f"Error in report generation: {str(e)}")
# #         raise

# import pandas as pd
# import dtale
# import os
# import logging

# logger = logging.getLogger(__name__)

# def report(csv_file):
#     """
#     Purpose: Generate a profiling report from a CSV file using D-Tale
#     """
#     logger.debug("Starting D-Tale report generation")
    
#     try:
#         df = pd.read_csv(csv_file)
#         logger.info(f"CSV file {csv_file.name} successfully loaded")
        
#         # Launch D-Tale instance
#         d = dtale.show(df)
#         dtale_url = d._main_url  # Get the URL for visualization
        
#         logger.info(f"D-Tale report available at: {dtale_url}")
#         return dtale_url  # Return the URL for Django to use
        
#     except Exception as e:
#         logger.error(f"Error in report generation: {str(e)}")
#         raise
