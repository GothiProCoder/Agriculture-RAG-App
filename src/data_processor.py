import pdfplumber
import pandas as pd
import re
from typing import Tuple

def process_contact_info(raw_text: str) -> Tuple[str, str]:
    """
    Parses raw contact strings to separate names from phone numbers.
    
    Args:
        raw_text (str): The raw cell content containing name and/or numbers.
        
    Returns:
        Tuple[str, str]: Cleaned Name, Cleaned Phone Number(s).
    """
    if not raw_text:
        return "Not Available", "Not Available"

    clean_digits_text = raw_text.replace('.', '').replace('-', '').replace(' ', '')
    phone_matches = re.findall(r'\d{10}', clean_digits_text)
    final_phone = ", ".join(sorted(set(phone_matches)))

    if not final_phone:
        final_phone = "Not Available"

    name_text = re.sub(r'[\d\.\-\_]', ' ', raw_text)
    name_text = " ".join(name_text.split())

    if len(name_text) < 3:
        name_text = "Not Available"

    return name_text, final_phone

def parse_gaushala_pdf(pdf_file) -> pd.DataFrame:
    """
    Extracts tabular data from the uploaded PDF file and cleans specific columns.
    
    Args:
        pdf_file: The file object from Streamlit uploader.
        
    Returns:
        pd.DataFrame: Cleaned and structured DataFrame.
    """
    data = []
    current_district = "Unknown"
    header_keywords = ["Sr. No.", "Goshala Name", "List of Registered", "Registratio"]

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table: continue

            for row in table:
                cleaned_row = [str(cell).strip().replace('\n', ' ') if cell else "" for cell in row]
                if not any(cleaned_row): continue

                row_text = " ".join(cleaned_row).lower()

                # Logic to determine if row is data or metadata/header
                has_cattle_data = False
                if len(cleaned_row) > 5:
                    if re.search(r'\d+', cleaned_row[5]): has_cattle_data = True
                    if cleaned_row[5] == '0' or "closed" in cleaned_row[4].lower(): has_cattle_data = True

                if any(k.lower() in row_text for k in header_keywords) and not has_cattle_data:
                    continue

                # District Section Header Detection
                if ("distt" in row_text or "district" in row_text) and not has_cattle_data:
                    match = re.search(r'(?:distt\.?|district)\s*([a-zA-Z\s\.]+)', row_text, re.IGNORECASE)
                    if match:
                        raw_dist = match.group(1).strip()
                        clean_dist = re.sub(r'(total|cattle|sr\.|\d+)', '', raw_dist, flags=re.IGNORECASE).strip().title()
                        if clean_dist.endswith('.'): clean_dist = clean_dist[:-1]
                        if len(clean_dist) > 1:
                            current_district = clean_dist
                    continue

                if "total cattle" in row_text: continue
                if len(cleaned_row) < 3: continue

                try:
                    global_sr = cleaned_row[0]
                    if not global_sr.isdigit(): continue

                    distt_sr = cleaned_row[1] if len(cleaned_row) > 1 else ""
                    name = cleaned_row[2]
                    village = cleaned_row[3] if len(cleaned_row) > 3 else ""
                    reg_no = cleaned_row[4] if len(cleaned_row) > 4 else ""
                    cattle_raw = cleaned_row[5] if len(cleaned_row) > 5 else "0"
                    mobile_raw = cleaned_row[6] if len(cleaned_row) > 6 else ""

                    status = "Active"
                    if "Closed" in reg_no or "Closed" in cattle_raw or "(Closed)" in row_text:
                        status = "Closed"

                    cattle_count = re.sub(r'\D', '', cattle_raw)
                    cattle_count = int(cattle_count) if cattle_count else 0

                    contact_person, phone_number = process_contact_info(mobile_raw)

                    entry = {
                        "Global_Sr": global_sr,
                        "Distt_Sr": distt_sr,
                        "District": current_district,
                        "Gaushala_Name": name,
                        "Village": village,
                        "Registration_No": reg_no,
                        "Cattle_Count": cattle_count,
                        "Contact_Person": contact_person,
                        "Phone_Number": phone_number,
                        "Status": status
                    }
                    data.append(entry)

                except Exception:
                    continue

    return pd.DataFrame(data)