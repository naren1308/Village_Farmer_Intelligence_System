import pandas as pd
import glob
import os

def filter_massive_dataset():
    """
    Reads all the yearly CSV files inside data/raw/csv, filters them for 
    Tamil Nadu and extracts Tomato, Onion, and Groundnut separately.
    """
    input_path = "data/raw/csv/*.csv"
    
    all_files = glob.glob(input_path)
    if not all_files:
        print("No CSV files found in data/raw/csv/. Please re-unzip archive.zip!")
        return
        
    print(f"Found {len(all_files)} CSV files to process.")
    
    chunks = {'Tomato': [], 'Onion': [], 'Groundnut': []}
    
    # Process each file to avoid MemoryErrors
    for file in all_files:
        print(f"Processing {os.path.basename(file)}...")
        try:
            df = pd.read_csv(file, low_memory=False)
            if 'State' in df.columns and 'Commodity' in df.columns:
                df_tn = df[df['State'] == 'Tamil Nadu']
                
                # Tomato
                tomato_df = df_tn[df_tn['Commodity'].str.contains('Tomato', case=False, na=False)]
                if not tomato_df.empty: chunks['Tomato'].append(tomato_df)
                
                # Onion
                onion_df = df_tn[df_tn['Commodity'].str.contains('Onion', case=False, na=False)]
                if not onion_df.empty: chunks['Onion'].append(onion_df)
                
                # Groundnut
                gnut_df = df_tn[df_tn['Commodity'].str.contains('Groundnut', case=False, na=False)]
                if not gnut_df.empty: chunks['Groundnut'].append(gnut_df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    for crop, chunk_list in chunks.items():
        if chunk_list:
            final_df = pd.concat(chunk_list, ignore_index=True)
            if 'Arrival_Date' in final_df.columns:
                final_df.rename(columns={'Arrival_Date': 'Date'}, inplace=True)
            final_df['Date'] = pd.to_datetime(final_df['Date'])
            final_df = final_df.sort_values('Date').reset_index(drop=True)
            
            output_file = f"data/raw/agmarknet_78_TN_{crop}.csv"
            final_df.to_csv(output_file, index=False)
            print(f"Saved {len(final_df)} records for {crop} to {output_file}")

if __name__ == "__main__":
    filter_massive_dataset()
