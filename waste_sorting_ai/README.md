# 🗂️ AI Waste Sorting & Recycling System

An AI-powered image classification system for sorting waste into recyclable and non-recyclable categories using MobileNetV2 transfer learning.

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+ (tested on Python 3.13)
- Git
- 4GB+ RAM (for TensorFlow)
- Linux/macOS/Windows

### 1. Clone & Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd waste_sorting_ai

# Create virtual environment (REQUIRED)
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies (this takes 2-3 minutes)
pip install -r requirements.txt
```

⚠️ **IMPORTANT**: Always activate the virtual environment before running any commands!

### 2. Verify Installation
```bash
# Test the system
python demo.py
```
You should see: ✅ Dataset loaded, ⚠️ No trained model found (expected)

### 3. Add Real Waste Images

**DO THIS for better accuracy:**
```bash
# Download 15-20 images from Google Images for each category:

# Recyclable images (save as .jpg):
# - plastic_bottle1.jpg, plastic_bottle2.jpg, etc.
# - aluminum_can1.jpg, aluminum_can2.jpg, etc.
# - glass_bottle1.jpg, cardboard1.jpg, etc.

# Non-recyclable images:
# - banana_peel1.jpg, food_scraps1.jpg, etc.
# - dirty_container1.jpg, mixed_waste1.jpg, etc.

# Put ALL images in data/raw/ folder
cp your_downloaded_images/*.jpg data/raw/
```

### 4. Label Your Images
Edit `data/labels.csv` (replace sample data):
```csv
filename,category,recyclable,location,notes
plastic_bottle1.jpg,plastic_bottle,true,kitchen,clear water bottle
aluminum_can1.jpg,aluminum_can,true,kitchen,soda can
banana_peel1.jpg,organic,false,kitchen,banana peel
food_scraps1.jpg,organic,false,kitchen,mixed food waste
glass_bottle1.jpg,glass_bottle,true,kitchen,wine bottle
dirty_container1.jpg,mixed_waste,false,kitchen,contaminated container
```

### 5. Train the AI Model
```bash
# This takes 2-5 minutes depending on your CPU
python src/model_training.py
```

**What to expect:**
- ✅ "Loaded X images" - should be 15+ for good results
- ✅ "Training model..." - 20 epochs with progress bars
- ✅ "Model saved to models/waste_classifier.h5"
- ⚠️ Warnings about CUDA/protobuf are normal (not errors)

### 6. Test Your AI
```bash
# Launch web interface (recommended)
python src/web_app.py
```
- Opens at `http://localhost:7860`
- Drag & drop images to test
- Should see "Model loaded" (not "Model not loaded")

```bash
# OR test via command line
python src/prediction.py
# Commands: predict data/raw/your_image.jpg
```

## 🎯 For University Assignment Demo

### What to Show Your Professor:
1. **Web Interface**: `python src/web_app.py` - drag/drop demo
2. **System Overview**: `python demo.py`
3. **Report**: `cat report_template.md` - African context analysis
4. **Code Structure**: Show clean, documented code

### Expected Accuracy:
- With sample random images: ~50%
- With real waste images: 80-90%+ 

## 📁 Project Structure
```
waste_sorting_ai/
├── data/
│   ├── raw/                 # Your waste images go here
│   ├── processed/           # Auto-generated
│   └── labels.csv          # Image labels (edit this!)
├── models/
│   ├── waste_classifier.h5 # Trained AI model
│   └── training_history.png # Training plots
├── src/
│   ├── data_collection.py  # Data management
│   ├── model_training.py   # AI training script
│   ├── prediction.py       # Prediction engine
│   └── web_app.py         # Web interface
├── venv/                   # Virtual environment
├── demo.py                # Quick system test
├── report_template.md     # Assignment report
└── requirements.txt       # Dependencies
```

## ⚠️ Common Issues & Solutions

### "ModuleNotFoundError"
```bash
# Always activate virtual environment first!
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### "Model not loaded" in web app
```bash
# Train model first
python src/model_training.py
```

### "Too few samples" error
- Need at least 2 images per class
- Recommended: 15+ images per class for good accuracy

### Web app not opening
- Check the terminal for the local URL (usually `http://localhost:7860`)
- Try a different browser
- Check firewall settings

### TensorFlow warnings
- CUDA warnings are normal (means using CPU, not GPU)
- Protobuf warnings are harmless
- Only worry about actual ERROR messages

## 🔧 Development Notes

### File Naming Convention:
- Images: `category_number.jpg` (e.g., `plastic_bottle1.jpg`)
- Keep filenames simple (no spaces or special characters)

### Adding New Categories:
1. Add images to `data/raw/`
2. Update `data/labels.csv`
3. Retrain: `python src/model_training.py`

### Model Performance:
- Training time: 2-5 minutes on modern CPU
- Model size: ~9MB (good for mobile deployment)
- Inference: <100ms per image

## 📊 Assignment Checklist

- [ ] Real waste images collected (15+ per category)
- [ ] Labels.csv updated with proper categories
- [ ] Model trained with >80% accuracy
- [ ] Web interface working and demo-ready
- [ ] Report completed (African context analysis)
- [ ] Code committed to Git with clear history
- [ ] System tested on different image types

## 🆘 Getting Help

### If something breaks:
1. Check you're in virtual environment: `which python`
2. Run the demo: `python demo.py`
3. Check git status: `git status`
4. Look for actual ERROR messages (ignore warnings)

### Performance Tips:
- More diverse images = better accuracy
- Clear, well-lit photos work best
- Balance your dataset (equal recyclable/non-recyclable)

## 🎓 Academic Context

This system demonstrates:
- **Transfer Learning**: Using pre-trained MobileNetV2
- **Computer Vision**: Image classification pipeline
- **Web Deployment**: Gradio interface
- **African Context**: Environmental impact analysis
- **Best Practices**: Virtual environments, version control, documentation

Perfect for AI/ML university coursework and portfolio projects!