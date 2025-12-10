"""Standalone script to run copresence analysis on cleaned RFID data.

Usage:
    python run_copresence_analysis.py
    
This script analyzes copresence patterns between chickens in nests
using the cleaned CSV files from the threshold_pre20s_post20s folder.
"""

from pathlib import Path
import sys

# Ensure package root is on sys.path when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rfid_analyzer import create_copresence_matrix


def main() -> None:
    """Run copresence analysis on all CSV files in the cleaned data folder."""
    
    # Hardcoded configuration
    # Input folder with cleaned CSV files
    input_folder = Path(r"..\..\cleaned\threshold_pre20s_post20s")
    
    # Output folder for copresence matrices
    output_folder = Path(r"..\..\output\copresence_matrices")
    
    # Analysis mode: 'count' or 'duration'
    # - 'count': counts the number of copresence events
    # - 'duration': sums the total seconds chickens were together
    mode = "count"  # Change to "count" for event counting
    
    # Convert to absolute paths
    input_folder = (Path(__file__).parent / input_folder).resolve()
    output_folder = (Path(__file__).parent / output_folder).resolve()
    
    # Verify input folder exists
    if not input_folder.exists():
        print(f"Error: Input folder not found: {input_folder}")
        return
    
    # Find all CSV files
    csv_files = sorted(input_folder.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in: {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Mode: {mode}")
    print(f"Output folder: {output_folder}")
    print("-" * 60)
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        try:
            create_copresence_matrix(
                csv_path=csv_file,
                output_folder=output_folder,
                mode=mode
            )
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_folder}")


if __name__ == "__main__":
    main()
