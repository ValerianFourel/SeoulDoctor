# --- 2. HELPER FUNCTIONS ---
import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two GPS coordinates."""
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def fuzzy_match_location(user_text: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuzzy match user's location text against file_district (roman) and file_dong (Hangul).
    Returns filtered dataframe.
    """
    if not user_text or len(df) == 0:
        return df
    
    user_lower = user_text.lower().strip()
    user_clean = user_lower.replace('gu', '').replace('dong', '').replace('-', '').strip()
    
    print(f"🔍 Fuzzy matching location: '{user_text}'")
    
    matched_rows = []
    
    try:
        for idx, row in df.iterrows():
            score = 0
            
            # Match against file_district
            if 'file_district' in row.index and pd.notna(row['file_district']):
                district = str(row['file_district']).lower().replace('-', '')
                district_clean = district.replace('gu', '').strip()
                
                if user_clean in district_clean or district_clean in user_clean:
                    score += 10
                elif any(word in district_clean for word in user_clean.split() if len(word) > 2):
                    score += 5
            
            # Match against file_dong
            if 'file_dong' in row.index and pd.notna(row['file_dong']):
                dong = str(row['file_dong'])
                
                if user_text in dong or dong in user_text:
                    score += 10
                elif any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in user_text):
                    if any(word in dong for word in user_text.split() if len(word) > 1):
                        score += 5
            
            # Match against address
            if 'address' in row.index and pd.notna(row['address']):
                address = str(row['address']).lower()
                if user_clean in address:
                    score += 3
            
            if score > 0:
                matched_rows.append((idx, score))
        
        if matched_rows:
            matched_rows.sort(key=lambda x: x[1], reverse=True)
            matched_indices = [idx for idx, score in matched_rows]
            result_df = df.loc[matched_indices]
            print(f"   ✅ Found {len(result_df)} facilities matching location")
            return result_df
        else:
            print(f"   ⚠️ No location matches found, using all facilities")
            return df
            
    except Exception as e:
        print(f"   ❌ Error in fuzzy matching: {e}, returning all facilities")
        return df
