import csv
import json
import pandas as pd
from collections import defaultdict
def convert_excel_to_json(excel_file_path, json_file_path):
    try:
        # 1. Read the Excel file directly
        df = pd.read_excel(excel_file_path)
        
        # Clean up column names (removes extra spaces like "Themes ")
        df.columns = df.columns.str.strip()

        # 2. Group verses by theme
        # Drop rows where Theme or Verse is missing
        df = df.dropna(subset=['Themes', 'Verses']) 
        
        # Group by theme and convert verses to a list
        grouped = df.groupby('Themes')['Verses'].apply(lambda x: [str(v).strip() for v in x]).to_dict()

        # 3. Format the data into the requested JSON structure
        final_json_data = []
        for serial_no, (theme, verses) in enumerate(grouped.items(), start=1):
            final_json_data.append({
                "serial_no": serial_no,
                "theme": theme,
                "verses": verses
            })

        # 4. Write to JSON
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            json.dump(final_json_data, json_file, indent=4, ensure_ascii=False)
            
        print(f"Success! Data successfully saved to '{json_file_path}'")

    except FileNotFoundError:
        print(f"Error: The Excel file '{excel_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
def convert_csv_to_json(csv_file_path, json_file_path):
    # This dictionary will automatically create a list for any new theme
    theme_verses = defaultdict(list)

    # 1. Read the CSV file
    try:
        # utf-8-sig handles any hidden characters at the start of the file
        with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            
            # Clean up header names just in case there are trailing spaces (like "Themes ")
            headers = [header.strip() if header else '' for header in reader.fieldnames]
            reader.fieldnames = headers

            # 2. Group verses by theme
            for row in reader:
                theme = row.get('Themes', '').strip()
                verse = row.get('Verses', '').strip()
                
                # Only add if both theme and verse are present
                if theme and verse:
                    theme_verses[theme].append(verse)

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    # 3. Format the data into the requested JSON structure
    final_json_data = []
    
    # enumerate provides an automatic serial number starting from 1
    for serial_no, (theme, verses) in enumerate(theme_verses.items(), start=1):
        final_json_data.append({
            "serial_no": serial_no,
            "theme": theme,
            "verses": verses
        })

    # 4. Write the structured data to a JSON file
    try:
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            json.dump(final_json_data, json_file, indent=4, ensure_ascii=False)
        print(f"Success! Data successfully saved to '{json_file_path}'")
    except Exception as e:
        print(f"An error occurred while writing the JSON: {e}")

# --- Execute the script ---
if __name__ == "__main__":
    # Update these filenames if yours are named differently
    input_excel = r'D:\ugp\Themes.xlsx'
    output_json = r'D:\ugp\themes_verses.json'
    
    convert_excel_to_json(input_excel, output_json)