#!/usr/bin/env python3
"""
Data Collection and Labeling System for Waste Classification
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image

class WasteDataCollector:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.labels_file = self.data_dir / "labels.csv"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize labels CSV
        if not self.labels_file.exists():
            with open(self.labels_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'category', 'recyclable', 'location', 'notes'])
    
    def add_image_label(self, filename: str, category: str, recyclable: bool, 
                       location: str = "", notes: str = ""):
        """Add a labeled image entry to the dataset"""
        with open(self.labels_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, category, recyclable, location, notes])
        print(f"Added label: {filename} -> {category} ({'recyclable' if recyclable else 'non-recyclable'})")
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for model training"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the complete dataset"""
        images = []
        labels = []
        
        with open(self.labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = self.raw_dir / row['filename']
                if image_path.exists():
                    try:
                        img = self.preprocess_image(str(image_path))
                        images.append(img)
                        labels.append(1 if row['recyclable'].lower() == 'true' else 0)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset"""
        stats = {'total': 0, 'recyclable': 0, 'non_recyclable': 0, 'categories': {}}
        
        with open(self.labels_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats['total'] += 1
                if row['recyclable'].lower() == 'true':
                    stats['recyclable'] += 1
                else:
                    stats['non_recyclable'] += 1
                
                category = row['category']
                stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        return stats

def main():
    """Interactive data collection interface"""
    collector = WasteDataCollector()
    
    print("=== Waste Classification Data Collector ===")
    print("Commands:")
    print("  add <filename> <category> <recyclable> [location] [notes]")
    print("  stats - show dataset statistics")
    print("  quit - exit")
    
    while True:
        try:
            command = input("\n> ").strip().split()
            if not command:
                continue
                
            if command[0] == 'quit':
                break
            elif command[0] == 'stats':
                stats = collector.get_dataset_stats()
                print(f"Dataset Statistics:")
                print(f"  Total images: {stats['total']}")
                print(f"  Recyclable: {stats['recyclable']}")
                print(f"  Non-recyclable: {stats['non_recyclable']}")
                print(f"  Categories: {stats['categories']}")
            elif command[0] == 'add' and len(command) >= 4:
                filename = command[1]
                category = command[2]
                recyclable = command[3].lower() in ['true', 'yes', '1', 'recyclable']
                location = command[4] if len(command) > 4 else ""
                notes = " ".join(command[5:]) if len(command) > 5 else ""
                
                collector.add_image_label(filename, category, recyclable, location, notes)
            else:
                print("Invalid command. Use 'add <filename> <category> <recyclable>' or 'stats' or 'quit'")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Data collection session ended.")

if __name__ == "__main__":
    main()