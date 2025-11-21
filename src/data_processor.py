import pdfplumber
import pandas as pd
import re
from typing import Tuple, Optional, Any

def process_contact_info(raw_text: Any) -> Tuple[str, str]:
    """Parses raw contact strings to separate names from phone numbers."""
    if not raw_text:
        return None, None
    
    raw_text_str = str(raw_text)

    # Remove separators to find raw digits
    clean_digits_text = raw_text_str.replace('.', '').replace('-', '').replace(' ', '')
    phone_matches = re.findall(r'\d{10}', clean_digits_text)
    final_phone = ", ".join(sorted(set(phone_matches)))

    if not final_phone:
        final_phone = None

    # Remove digits and special chars to isolate name
    name_text = re.sub(r'[\d\.\-\_]', ' ', raw_text_str)
    name_text = " ".join(name_text.split())

    if len(name_text) < 3:
        name_text = None

    return name_text, final_phone

def enforce_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL: Enforces strict types for high-accuracy Pandas querying.
    """
    if df.empty:
        return df

    # 1. Force Cattle_Count to Int (Handle NaNs and non-numeric strings as 0)
    df['Cattle_Count'] = pd.to_numeric(df['Cattle_Count'], errors='coerce').fillna(0).astype(int)

    # 2. Normalize Strings (Title Case for consistency)
    text_cols = ['District', 'Gaushala_Name', 'Village', 'Status', 'Contact_Person', 'Phone_Number']
    for col in text_cols:
        if col in df.columns:
            # Convert to string, replace 'nan', strip whitespace, Title Case
            df[col] = df[col].apply(lambda x: x.strip().title() if isinstance(x, str) and x.lower() != 'nan' and x != '' else None)
            # Fill empty strings with reasonable defaults if needed
            df[col] = df[col].replace(["Not Available", "not available", "Unknown", "nan"], None)
    
    # 3. Ensure Registration No is string and consistent
    if 'Registration_No' in df.columns:
        df['Registration_No'] = df['Registration_No'].astype(str).str.strip()

    return df

def parse_gaushala_pdf(pdf_file) -> pd.DataFrame:
    """Extracts tabular data from the uploaded PDF file."""
    data = []
    current_district = "Unknown"
    header_keywords = ["Sr. No.", "Goshala Name", "List of Registered", "Registratio", "Name of"]

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if not table: continue

                for row in table:
                    # Clean None values to empty strings
                    cleaned_row = [str(cell).strip().replace('\n', ' ') if cell is not None else "" for cell in row]
                    if not any(cleaned_row): continue

                    row_text = " ".join(cleaned_row).lower()

                    # Logic to determine if row is data or metadata/header
                    has_cattle_data = False
                    # Heuristic: Column 5 often has the count in this specific PDF format
                    if len(cleaned_row) > 5:
                        if re.search(r'\d+', cleaned_row[5]): has_cattle_data = True
                        if cleaned_row[5] == '0' or "closed" in cleaned_row[4].lower(): has_cattle_data = True

                    # Skip Headers
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
                        # If first column isn't a digit, it's likely junk or header drift
                        if not global_sr.isdigit(): continue

                        distt_sr = cleaned_row[1] if len(cleaned_row) > 1 else ""
                        name = cleaned_row[2]
                        village = cleaned_row[3] if len(cleaned_row) > 3 else ""
                        reg_no = cleaned_row[4] if len(cleaned_row) > 4 else ""
                        cattle_raw = cleaned_row[5] if len(cleaned_row) > 5 else "0"
                        mobile_raw = cleaned_row[6] if len(cleaned_row) > 6 else ""

                        status = "Active"
                        if "closed" in reg_no.lower() or "closed" in cattle_raw.lower() or "(closed)" in row_text:
                            status = "Closed"
                            
                        # 2. Clean Registration Number
                        # Remove "(Closed)", "Closed", parens, and extra spaces
                        # This turns "GSA-312 (Closed)" -> "GSA-312"
                        reg_no = re.sub(r'\(?closed\)?', '', reg_no, flags=re.IGNORECASE).strip()

                        contact_person, phone_number = process_contact_info(mobile_raw)

                        entry = {
                            "Global_Sr": global_sr,
                            "Distt_Sr": distt_sr,
                            "District": current_district,
                            "Gaushala_Name": name,
                            "Village": village,
                            "Registration_No": reg_no,
                            "Cattle_Count": cattle_raw, # Processed in enforce_data_types
                            "Contact_Person": contact_person,
                            "Phone_Number": phone_number,
                            "Status": status
                        }
                        data.append(entry)

                    except Exception:
                        continue
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    return enforce_data_types(df)