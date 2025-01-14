from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
import logging
import os
from .my_report import report

# Get an instance of a logger
logger = logging.getLogger(__name__)

class IndexView(View):
    template_name = "myapp/index.html"

    def get(self, request):
        logger.debug("GET request received")
        return render(request, self.template_name)

    def post(self, request):
        logger.debug("POST request received")

        # Get uploaded CSV file
        csv_file = request.FILES.get('csv')
        if not csv_file:
            logger.error("No file uploaded")
            return HttpResponse("<h1>Error: No file uploaded!</h1>", status=400)

        try:
            # Call the report function to generate the profile
            logger.info("Generating report for the uploaded CSV file")
            report(csv_file)
            logger.info("Report generation successful")
            
            # If the report generation is successful, remove the old log file
            log_file_path = 'debug.log'
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
                logger.info("Old log file removed successfully")
            
        except Exception as e:
            # Log error if something goes wrong
            logger.error(f"Error during report generation: {str(e)}")
            return HttpResponse(f"<h1>Error: {str(e)}</h1>", status=500)
        
        # If successful, render the result
        return render(request, 'pandas_profiling_report.html')
