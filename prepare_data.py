"""
Utility script to prepare training data CSV from JSON annotations.

Usage:
    python prepare_data.py --json_path annotations.json --output_csv train.csv
"""

import json
import csv
import argparse
import os


def json_to_csv(json_path, output_csv):
    """
    Convert JSON annotations to CSV format.
    
    Expected JSON format:
    [
        {
            "filename": "img_00000.jpg",
            "original_path": "...",
            "width": 272,
            "height": 306,
            "words": [
                {
                    "text": "HP54C6564",
                    "xmin": 105,
                    "xmax": 167,
                    "ymin": 183,
                    "ymax": 197
                }
            ]
        },
        ...
    ]
    
    Output CSV format:
        filename,width,height,label,xmin,ymin,xmax,ymax
    """
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for item in data:
        filename = item['filename']
        width = item['width']
        height = item['height']
        
        # Get first word (number plate text)
        if item['words']:
            word = item['words'][0]
            label = word['text']
            xmin = word['xmin']
            xmax = word['xmax']
            ymin = word['ymin']
            ymax = word['ymax']
            
            rows.append({
                'filename': filename,
                'width': width,
                'height': height,
                'label': label,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Created {output_csv} with {len(rows)} entries")


def create_sample_csv():
    """
    Create a sample CSV file for testing.
    Uses the examples from dataset.txt
    """
    sample_data = [
        {'filename': 'img_00000.jpg', 'width': 272, 'height': 306, 'label': 'HP54C6564', 'xmin': 105, 'ymin': 183, 'xmax': 167, 'ymax': 197},
        {'filename': 'img_00001.jpg', 'width': 272, 'height': 204, 'label': 'HP302165', 'xmin': 81, 'ymin': 124, 'xmax': 131, 'ymax': 137},
        {'filename': 'img_00002.jpg', 'width': 191, 'height': 139, 'label': 'HP01A7630', 'xmin': 30, 'ymin': 97, 'xmax': 74, 'ymax': 111},
    ]
    
    os.makedirs('./data', exist_ok=True)
    
    with open('./data/train.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])
        writer.writeheader()
        writer.writerows(sample_data)
    
    print("Created sample ./data/train.csv")
    print("\nTo use the full dataset:")
    print("1. Place your images in ./data/images/")
    print("2. Run: python prepare_data.py --json_path your_annotations.json --output_csv ./data/train.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare training data CSV')
    parser.add_argument('--json_path', type=str, default=None,
                        help='Path to JSON annotations file')
    parser.add_argument('--output_csv', type=str, default='./data/train.csv',
                        help='Output CSV file path')
    parser.add_argument('--sample', action='store_true',
                        help='Create a sample CSV for testing')
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_csv()
    elif args.json_path:
        json_to_csv(args.json_path, args.output_csv)
    else:
        print("Usage:")
        print("  Create sample CSV:  python prepare_data.py --sample")
        print("  Convert JSON to CSV: python prepare_data.py --json_path annotations.json --output_csv train.csv")
