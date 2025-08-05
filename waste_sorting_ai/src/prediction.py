#!/usr/bin/env python3
"""
Prediction module for waste classification
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from pathlib import Path

class WastePredictor:
    def __init__(self, model_path='models/waste_classifier.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image_path_or_array, target_size=(224, 224)):
        """Preprocess image for prediction"""
        if isinstance(image_path_or_array, (str, Path)):
            # Load from file
            image = cv2.imread(str(image_path_or_array))
            if image is None:
                raise ValueError(f"Could not load image: {image_path_or_array}")
        else:
            # Use provided array
            image = image_path_or_array
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_path_or_array):
        """Make prediction on a single image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path_or_array)
        
        # Make prediction
        prediction_prob = self.model.predict(processed_image, verbose=0)[0][0]
        prediction_class = 1 if prediction_prob > 0.5 else 0
        
        # Format result
        result = {
            'recyclable': bool(prediction_class),
            'confidence': float(prediction_prob if prediction_class else 1 - prediction_prob),
            'raw_probability': float(prediction_prob),
            'class_name': 'Recyclable' if prediction_class else 'Non-Recyclable'
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """Make predictions on multiple images"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = str(image_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': str(image_path),
                    'error': str(e),
                    'recyclable': None,
                    'confidence': 0.0
                })
        
        return results

def main():
    """Interactive prediction interface"""
    predictor = WastePredictor()
    
    if predictor.model is None:
        print("No trained model found. Please train a model first.")
        return
    
    print("=== Waste Classification Predictor ===")
    print("Commands:")
    print("  predict <image_path> - predict single image")
    print("  batch <folder_path> - predict all images in folder")
    print("  quit - exit")
    
    while True:
        try:
            command = input("\n> ").strip().split()
            if not command:
                continue
            
            if command[0] == 'quit':
                break
            elif command[0] == 'predict' and len(command) == 2:
                image_path = command[1]
                result = predictor.predict(image_path)
                print(f"\nPrediction for {image_path}:")
                print(f"  Class: {result['class_name']}")
                print(f"  Confidence: {result['confidence']:.2%}")
                print(f"  Recyclable: {result['recyclable']}")
            elif command[0] == 'batch' and len(command) == 2:
                folder_path = Path(command[1])
                if not folder_path.exists():
                    print(f"Folder not found: {folder_path}")
                    continue
                
                # Find all image files
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(folder_path.glob(f'*{ext}'))
                    image_files.extend(folder_path.glob(f'*{ext.upper()}'))
                
                if not image_files:
                    print(f"No image files found in {folder_path}")
                    continue
                
                print(f"\nProcessing {len(image_files)} images...")
                results = predictor.predict_batch(image_files)
                
                print("\nBatch Predictions:")
                recyclable_count = 0
                for result in results:
                    if 'error' in result:
                        print(f"  {result['image_path']}: ERROR - {result['error']}")
                    else:
                        print(f"  {result['image_path']}: {result['class_name']} ({result['confidence']:.1%})")
                        if result['recyclable']:
                            recyclable_count += 1
                
                print(f"\nSummary: {recyclable_count} recyclable, {len(results) - recyclable_count} non-recyclable")
            else:
                print("Invalid command. Use 'predict <image_path>' or 'batch <folder_path>' or 'quit'")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Prediction session ended.")

if __name__ == "__main__":
    main()