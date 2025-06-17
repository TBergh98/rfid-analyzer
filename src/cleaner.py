import os
import pandas as pd
from datetime import datetime, timedelta
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_datetime(date_str, time_str):
    """Parse date and time strings into datetime object"""
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M:%S")
    except ValueError as e:
        logger.warning(f"Invalid datetime format: {date_str} {time_str} - {e}")
        return None

def is_valid_row(row):
    """Check if a row contains valid data"""
    # Check if all required fields are present and not empty
    if len(row) < 5:
        return False
    
    date, time, action, chicken_id, nest_id = row[:5]
    
    # Check for empty or null values
    if pd.isna(date) or pd.isna(time) or pd.isna(action) or pd.isna(chicken_id) or pd.isna(nest_id):
        return False
    
    # Check for empty strings
    if not str(date).strip() or not str(time).strip() or not str(action).strip():
        return False
    
    # Check if action is valid
    if str(action).strip().upper() not in ['IN', 'OUT']:
        return False
    
    # Check if chicken_id and nest_id are valid (should be numeric or valid identifiers)
    try:
        # Try to convert to string and check if not empty after stripping
        chicken_id_str = str(chicken_id).strip()
        nest_id_str = str(nest_id).strip()
        if not chicken_id_str or not nest_id_str:
            return False
    except:
        return False
    
    return True

def clean_chicken_data(df, threshold_seconds):
    """Clean chicken data by merging short exits/re-entries"""
    if df.empty:
        logger.warning("Empty dataframe received for cleaning")
        return df
    
    df_sorted = df.sort_values(['chicken_id', 'nest_id', 'datetime']).reset_index(drop=True)
    
    cleaned_data = []
    
    for (chicken_id, nest_id), group in df_sorted.groupby(['chicken_id', 'nest_id']):
        group = group.reset_index(drop=True)
        i = 0
        
        while i < len(group):
            current_row = group.iloc[i]
            
            # Look for pattern: IN -> OUT -> IN within threshold
            if (current_row['action'] == 'IN' and 
                i + 2 < len(group) and 
                group.iloc[i + 1]['action'] == 'OUT' and 
                group.iloc[i + 2]['action'] == 'IN'):
                
                out_time = group.iloc[i + 1]['datetime']
                in_time = group.iloc[i + 2]['datetime']
                
                if (in_time - out_time).total_seconds() <= threshold_seconds:
                    # Skip the OUT and second IN, keep only the first IN
                    cleaned_data.append(current_row)
                    i += 3
                else:
                    cleaned_data.append(current_row)
                    i += 1
            else:
                cleaned_data.append(current_row)
                i += 1
    
    return pd.DataFrame(cleaned_data)

def process_csv_file(file_path, threshold_seconds):
    """Process a single CSV file with robust error handling"""
    try:
        # Read file with error handling
        df = pd.read_csv(file_path, sep=';', header=None, 
                        names=['date', 'time', 'action', 'chicken_id', 'nest_id'],
                        on_bad_lines='skip',  # Skip malformed lines
                        engine='python')     # Use python engine for better error handling
        
        logger.info(f"Read {len(df)} rows from {file_path}")
        
        # Filter out corrupted/invalid rows
        valid_rows = []
        corrupted_count = 0
        
        for idx, row in df.iterrows():
            if is_valid_row(row.values):
                valid_rows.append(idx)
            else:
                corrupted_count += 1
                logger.debug(f"Skipping corrupted row {idx}: {row.values}")
        
        if corrupted_count > 0:
            logger.info(f"Filtered out {corrupted_count} corrupted rows from {file_path}")
        
        # Keep only valid rows
        df = df.loc[valid_rows].reset_index(drop=True)
        
        if df.empty:
            logger.warning(f"No valid data found in {file_path}")
            return pd.DataFrame(columns=['date', 'time', 'action', 'chicken_id', 'nest_id'])
        
        # Parse datetime with error handling
        valid_datetime_rows = []
        datetime_errors = 0
        
        for idx, row in df.iterrows():
            parsed_datetime = parse_datetime(row['date'], row['time'])
            if parsed_datetime is not None:
                df.at[idx, 'datetime'] = parsed_datetime
                valid_datetime_rows.append(idx)
            else:
                datetime_errors += 1
        
        if datetime_errors > 0:
            logger.info(f"Filtered out {datetime_errors} rows with invalid datetime from {file_path}")
        
        # Keep only rows with valid datetime
        df = df.loc[valid_datetime_rows].reset_index(drop=True)
        
        if df.empty:
            logger.warning(f"No rows with valid datetime found in {file_path}")
            return pd.DataFrame(columns=['date', 'time', 'action', 'chicken_id', 'nest_id'])
        
        # Normalize action values
        df['action'] = df['action'].str.strip().str.upper()
        
        # Clean the data
        cleaned_df = clean_chicken_data(df, threshold_seconds)
        
        # Remove the datetime column and return original format
        cleaned_df = cleaned_df.drop('datetime', axis=1)
        
        logger.info(f"Successfully processed {file_path}: {len(cleaned_df)} valid rows")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        # Return empty dataframe instead of raising exception
        return pd.DataFrame(columns=['date', 'time', 'action', 'chicken_id', 'nest_id'])

def clean_chicken_csv_files(raw_input_folder, cleaned_output_folder, artefact_threshold_seconds):
    """
    Wrapper function to clean all chicken CSV files in a folder.
    
    Args:
        raw_input_folder (str): Input folder containing raw CSV files
        cleaned_output_folder (str): Output folder for cleaned CSV files
        artefact_threshold_seconds (int): Threshold in seconds for merging short exits
    
    Returns:
        bool: True if successful (at least one output file created), False if critical failure
    """
    try:
        # Validate input folder
        if not os.path.exists(raw_input_folder):
            logger.error(f"Input folder does not exist: {raw_input_folder}")
            return False
        
        # Create output subfolder with threshold suffix
        output_subfolder = os.path.join(cleaned_output_folder, f"threshold_{artefact_threshold_seconds}s")
        
        # Check if output subfolder already exists
        if os.path.exists(output_subfolder):
            # Check if it contains CSV files
            existing_csv_files = [f for f in os.listdir(output_subfolder) if f.endswith('.csv')]
            if existing_csv_files:
                logger.info(f"Output subfolder 'threshold_{artefact_threshold_seconds}s' already exists with {len(existing_csv_files)} CSV files.")
                logger.info("Skipping data cleaning step. You can proceed with the analysis.")
                return True
            else:
                logger.info(f"Output subfolder 'threshold_{artefact_threshold_seconds}s' exists but is empty. Proceeding with cleaning...")
        else:
            try:
                os.makedirs(output_subfolder, exist_ok=True)
                logger.info(f"Created output subfolder: threshold_{artefact_threshold_seconds}s")
            except Exception as e:
                logger.error(f"Failed to create output folder: {e}")
                return False
        
        # Process all CSV files in input folder
        csv_files = [f for f in os.listdir(raw_input_folder) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning("No CSV files found in input folder.")
            return False
        
        successful_files = 0
        total_files = len(csv_files)
        
        for csv_file in csv_files:
            input_path = os.path.join(raw_input_folder, csv_file)
            output_path = os.path.join(output_subfolder, csv_file)
            
            try:
                logger.info(f"Processing {csv_file}...")
                cleaned_df = process_csv_file(input_path, artefact_threshold_seconds)
                
                # Always try to save the file, even if empty
                cleaned_df.to_csv(output_path, sep=';', header=False, index=False)
                
                if not cleaned_df.empty:
                    logger.info(f"Saved cleaned data to {output_path} ({len(cleaned_df)} rows)")
                else:
                    logger.warning(f"Saved empty file to {output_path} (no valid data)")
                
                successful_files += 1
                
            except Exception as e:
                logger.error(f"Critical error processing {csv_file}: {str(e)}")
                # Continue processing other files instead of stopping
                continue
        
        # Check if we managed to create at least some output files
        created_files = len([f for f in os.listdir(output_subfolder) if f.endswith('.csv')])
        
        if created_files == 0:
            logger.error("No output files were created!")
            return False
        
        logger.info(f"Data cleaning completed. Created {created_files}/{total_files} output files in 'threshold_{artefact_threshold_seconds}s' folder")
        
        if created_files < total_files:
            logger.warning(f"Some files failed to process ({total_files - created_files} failures)")
        
        # Return True if at least one file was processed successfully
        return True
        
    except Exception as e:
        logger.error(f"Critical error in data cleaning process: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Clean chicken nesting data')
    parser.add_argument('raw_input_folder', help='Input folder containing raw CSV files')
    parser.add_argument('cleaned_output_folder', help='Output folder for cleaned CSV files')
    parser.add_argument('artefact_threshold_seconds', type=int, help='Threshold in seconds for merging short exits')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = clean_chicken_csv_files(args.raw_input_folder, args.cleaned_output_folder, args.artefact_threshold_seconds)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()