"""
Prediction script for CT/MRI classifier
Use a trained model to classify new brain scan images
"""

import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime

class CTMRIPredictor:
    """Make predictions using trained CT/MRI classifier"""
    
    def __init__(self, model_path):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model (.h5 file)
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(str(self.model_path))
        print("✓ Model loaded successfully")
        
        self.class_names = ["CT", "MRI"]
    
    def predict_single_image(self, image_path, verbose=True):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            verbose: Print prediction details
        
        Returns:
            Dictionary with prediction results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            # Interpret prediction
            class_idx = 1 if prediction > 0.5 else 0
            class_name = self.class_names[class_idx]
            confidence = prediction if prediction > 0.5 else (1 - prediction)
            
            result = {
                "filename": image_path.name,
                "predicted_class": class_name,
                "confidence": float(confidence),
                "raw_prediction": float(prediction),
                "timestamp": datetime.now().isoformat()
            }
            
            if verbose:
                print(f"\n{'='*50}")
                print(f"Image: {image_path.name}")
                print(f"Predicted: {class_name}")
                print(f"Confidence: {confidence:.2%}")
                print(f"{'='*50}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            return None
    
    def predict_batch(self, image_dir, output_file=None, verbose=True):
        """
        Predict class for all images in a directory
        
        Args:
            image_dir: Directory containing images
            output_file: Optional JSON file to save results
            verbose: Print prediction details
        
        Returns:
            List of prediction results
        """
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # Find all image files
        image_files = list(image_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in 
                      ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return []
        
        print(f"\nProcessing {len(image_files)} images from {image_dir}")
        print("="*50)
        
        results = []
        ct_count = 0
        mri_count = 0
        
        for img_path in image_files:
            result = self.predict_single_image(img_path, verbose=False)
            
            if result:
                results.append(result)
                
                if result["predicted_class"] == "CT":
                    ct_count += 1
                else:
                    mri_count += 1
                
                # Print summary
                confidence_pct = f"{result['confidence']:.1%}"
                print(f"  {result['filename']:<40} → {result['predicted_class']:<4} ({confidence_pct})")
        
        print("="*50)
        print(f"\nSummary:")
        print(f"  Total images: {len(results)}")
        print(f"  CT scans: {ct_count}")
        print(f"  MRI scans: {mri_count}")
        
        # Save results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ Results saved to {output_path}")
        
        return results
    
    def predict_with_visualization(self, image_path, output_path=None):
        """
        Predict and create visualization with prediction overlay
        
        Args:
            image_path: Path to image file
            output_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return None
        
        image_path = Path(image_path)
        
        # Get prediction
        result = self.predict_single_image(image_path, verbose=False)
        
        if not result:
            return None
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img)
        ax.axis('off')
        
        # Add prediction text
        title = f"{result['predicted_class']} ({result['confidence']:.1%})"
        color = 'green' if result['predicted_class'] == 'CT' else 'blue'
        ax.set_title(title, fontsize=16, fontweight='bold', color=color, pad=20)
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {output_path}")
        
        plt.show()
        
        return result


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Predict CT/MRI class for brain scan images"
    )
    parser.add_argument(
        "model",
        help="Path to trained model (.h5 file)"
    )
    parser.add_argument(
        "--image",
        help="Path to single image for prediction"
    )
    parser.add_argument(
        "--batch",
        help="Path to directory of images for batch prediction"
    )
    parser.add_argument(
        "--output",
        help="Output file for batch predictions (JSON)"
    )
    parser.add_argument(
        "--visualize",
        help="Create visualization for single image"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = CTMRIPredictor(args.model)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    
    # Single image prediction
    if args.image:
        try:
            result = predictor.predict_single_image(args.image)
            if not result:
                return 1
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
    
    # Batch prediction
    elif args.batch:
        try:
            predictor.predict_batch(args.batch, output_file=args.output)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
    
    # Visualization
    elif args.visualize:
        try:
            predictor.predict_with_visualization(args.visualize)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
