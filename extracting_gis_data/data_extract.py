import os
import re
import pandas as pd

DATA_DIR = "/home/udit/Documents/Github/ISSA/extracting_gis_data"  # Folder where 2016.txt ... 2024.txt exist
OUTPUT_FILE = "landslide_data.xlsx"

month_names = [
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"
]

def extract_entries(text, year):
    entries = []
    current_month = None
    blocks = re.split(r'\n(?=' + '|'.join(month_names) + r')', text)

    for block in blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue

        if any(month in lines[0] for month in month_names):
            current_month = lines[0].strip()
            continue

        match = re.match(r"(.+?)\s+on\s+(\d{1,2}(st|nd|rd|th)?\s+\w+\s+\d{4})", lines[0])
        if match:
            place = match.group(1).strip()
        else:
            place = lines[0].strip()

        description = "\n".join(lines[1:]).strip()
        if not description:
            continue

        location_matches = re.findall(r'([0-9]{1,2}\.?[0-9]*°?[ \']?[0-9]*[ \']?[0-9]*\.?\d*[”"]?\s?[NS])[,; ]+([0-9]{1,3}\.?[0-9]*°?[ \']?[0-9]*[ \']?[0-9]*\.?\d*[”"]?\s?[EW])', description)
        coordinates = "; ".join([f"{lat}, {lon}" for lat, lon in location_matches]) if location_matches else ""

        entries.append({
            "Year": year,
            "Month": current_month,
            "Place": place,
            "Description": description,
            "Location (Lat, Long)": coordinates
        })

    return entries

def process_all_txt_files():
    all_entries = []

    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith(".txt"):
            year = filename.split('.')[0]
            with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                entries = extract_entries(text, year)
                all_entries.extend(entries)

    df = pd.DataFrame(all_entries)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Excel file created: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_all_txt_files()
