import os
import shutil
from bs4 import BeautifulSoup

# Define the paths
html_file = "/home/udit/Downloads/time_series_report.html"  # Change this to your actual file name
output_folder = "extracted_images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the HTML file
with open(html_file, "r", encoding="utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")

# Find all images
images = soup.find_all("img")

# Extract images
for i, img in enumerate(images):
    img_src = img.get("src")
    
    if img_src and img_src.startswith("data:image"):  # Check if image is base64
        img_data = img_src.split(",")[1]  # Extract base64 data
        img_bytes = base64.b64decode(img_data)
        
        img_path = os.path.join(output_folder, f"image_{i+1}.png")
        
        # Save image
        with open(img_path, "wb") as img_file:
            img_file.write(img_bytes)

        print(f"Saved: {img_path}")

print("✅ All images extracted and saved in:", output_folder)
