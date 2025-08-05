#!/usr/bin/env python3
"""
Quick demo script for the waste classification system
"""

import os
import sys
from pathlib import Path

def main():
    print("🗂️  AI Waste Sorting & Recycling System Demo")
    print("=" * 50)
    
    # Check if virtual environment is active
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Please activate the virtual environment first:")
        print("   source venv/bin/activate")
        return
    
    try:
        from src.data_collection import WasteDataCollector
        from src.prediction import WastePredictor
        
        # Test data collection
        print("\n1. Testing Data Collection System...")
        collector = WasteDataCollector()
        stats = collector.get_dataset_stats()
        print(f"   ✅ Dataset loaded: {stats['total']} images")
        print(f"   ✅ Recyclable: {stats['recyclable']}, Non-recyclable: {stats['non_recyclable']}")
        print(f"   ✅ Categories: {list(stats['categories'].keys())}")
        
        # Test prediction system (without trained model)
        print("\n2. Testing Prediction System...")
        predictor = WastePredictor()
        if predictor.model is None:
            print("   ⚠️  No trained model found (this is expected)")
            print("   ℹ️  To train a model, run: python src/model_training.py")
        else:
            print("   ✅ Model loaded successfully")
        
        print("\n3. System Components:")
        components = [
            ("Data Collection", "src/data_collection.py", "✅"),
            ("Model Training", "src/model_training.py", "✅"),
            ("Prediction", "src/prediction.py", "✅"),
            ("Web Interface", "src/web_app.py", "✅"),
            ("Sample Data", "data/labels.csv", "✅"),
            ("Report Template", "report_template.md", "✅")
        ]
        
        for name, path, status in components:
            if Path(path).exists():
                print(f"   {status} {name}: {path}")
            else:
                print(f"   ❌ {name}: {path} (missing)")
        
        print("\n4. Next Steps:")
        print("   📸 Add real images: python src/data_collection.py")
        print("   🧠 Train model: python src/model_training.py")
        print("   🌐 Launch web app: python src/web_app.py")
        print("   📊 View report: cat report_template.md")
        
        print("\n✨ Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()