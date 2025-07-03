#!/usr/bin/env python3
import pandas as pd
"""
Convert repurposing_samples.txt to CSV format.
Filters out header lines (starting with !) and converts tab-separated data to CSV.
"""

import csv
import sys
from pathlib import Path


def convert_txt_to_csv(input_file, output_file):
    """Convert tab-separated text file to CSV format."""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        # Read lines and filter out header comments
        data_lines = []
        for line in infile:
            line = line.strip()
            if line and not line.startswith('!'):
                data_lines.append(line)
        
        if not data_lines:
            print("No data lines found in file")
            return
        
        # Parse the first line to get headers
        headers = data_lines[0].split('\t')
        # Write CSV
        writer = csv.writer(outfile)
        writer.writerow(headers)
        
        # Write data rows
        for line in data_lines[1:]:
            row = line.split('\t')
            writer.writerow(row)


def main():
    input_file = "processed_datasets/repurposing_samples.txt"
    output_file = "processed_datasets/repurposing_samples.csv"
    
    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    
    convert_txt_to_csv(input_file, output_file)
    print(f"Converted {input_file} to {output_file}")

    df = pd.read_csv(output_file)
    print(df.head())
    print(df.shape)
    print("Removing duplicates")
    df = df.drop_duplicates(subset=['pert_iname'])
    print(df.shape)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main() 