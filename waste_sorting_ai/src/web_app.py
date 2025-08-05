#!/usr/bin/env python3
"""
Web interface for waste classification using Gradio
"""

import gradio as gr
import numpy as np
from PIL import Image
from prediction import WastePredictor
import os

class WasteClassificationApp:
    def __init__(self):
        self.predictor = WastePredictor()
        self.setup_interface()
    
    def classify_waste(self, image):
        """Classify uploaded waste image"""
        if image is None:
            return "Please upload an image", 0.0, "N/A"
        
        try:
            # Convert PIL image to numpy array
            image_array = np.array(image)
            
            # Make prediction
            result = self.predictor.predict(image_array)
            
            # Format output
            classification = result['class_name']
            confidence = result['confidence']
            recyclable_status = "✅ Recyclable" if result['recyclable'] else "❌ Non-Recyclable"
            
            # Create detailed message
            message = f"""
            **Classification:** {classification}
            **Status:** {recyclable_status}
            **Confidence:** {confidence:.1%}
            
            {'This item can be recycled! Please dispose of it in the recycling bin.' if result['recyclable'] else 'This item cannot be recycled. Please dispose of it in the general waste bin.'}
            """
            
            return message, confidence, recyclable_status
            
        except Exception as e:
            return f"Error processing image: {str(e)}", 0.0, "Error"
    
    def setup_interface(self):
        """Setup Gradio interface"""
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .output-class {
            font-size: 1.2em;
            font-weight: bold;
        }
        """
        
        # Create interface
        with gr.Blocks(css=css, title="AI Waste Sorting System") as self.interface:
            gr.Markdown("""
            # 🗂️ AI Waste Sorting & Recycling System
            
            Upload an image of waste to classify it as recyclable or non-recyclable.
            This system uses AI to help improve waste management and recycling efforts.
            """)
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Waste Image",
                        height=300
                    )
                    
                    classify_btn = gr.Button(
                        "🔍 Classify Waste",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column():
                    result_output = gr.Markdown(
                        label="Classification Result",
                        elem_classes=["output-class"]
                    )
                    
                    confidence_output = gr.Number(
                        label="Confidence Score",
                        precision=3
                    )
                    
                    status_output = gr.Textbox(
                        label="Recyclable Status",
                        interactive=False
                    )
            
            # Example images section
            gr.Markdown("## 📋 Example Images")
            gr.Markdown("Try these example waste items:")
            
            example_images = self.get_example_images()
            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=image_input,
                    outputs=[result_output, confidence_output, status_output],
                    fn=self.classify_waste,
                    cache_examples=True
                )
            
            # Connect button to function
            classify_btn.click(
                fn=self.classify_waste,
                inputs=image_input,
                outputs=[result_output, confidence_output, status_output]
            )
            
            # Footer
            gr.Markdown("""
            ---
            **About this system:**
            - Uses transfer learning with MobileNetV2 for efficient classification
            - Trained on local waste images for better accuracy in African contexts
            - Helps promote proper waste sorting and recycling practices
            
            **Note:** This is a demonstration system. For best results, ensure images are clear and well-lit.
            """)
    
    def get_example_images(self):
        """Get example images if they exist"""
        example_dir = "data/examples"
        if not os.path.exists(example_dir):
            return []
        
        examples = []
        for file in os.listdir(example_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                examples.append(os.path.join(example_dir, file))
        
        return examples[:5]  # Limit to 5 examples
    
    def launch(self, share=False, debug=True):
        """Launch the web application"""
        if self.predictor.model is None:
            print("⚠️  Warning: No trained model found!")
            print("Please train a model first using: python src/model_training.py")
            print("The app will still launch but predictions will not work.")
        
        print("🚀 Launching AI Waste Sorting Web Application...")
        print("📱 The app will be available in your browser")
        
        self.interface.launch(
            share=share,
            debug=debug,
            server_name="0.0.0.0",
            server_port=7860
        )

def main():
    """Launch the web application"""
    app = WasteClassificationApp()
    app.launch(share=True)

if __name__ == "__main__":
    main()