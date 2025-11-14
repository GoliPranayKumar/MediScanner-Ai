"""
Lightweight Medical Imaging Analysis
Uses PIL for image processing without heavy dependencies
No OpenCV dependency to avoid libGL issues in production
"""

import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class MedicalImagingAnalyzer:
    """
    Lightweight medical image analyzer
    Analyzes image properties without heavy ML models
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        self.model_loaded = True
    
    def analyze_image(self, image_path):
        """
        Analyze medical image and return findings
        """
        try:
            img_array = None
            
            # Try PIL first
            try:
                img = Image.open(image_path)
                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if img.mode not in ('RGB', 'L', 'RGBA'):
                    img = img.convert('RGB')
                elif img.mode == 'RGBA':
                    # Convert RGBA to RGB
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
                    img = rgb_img
                
                img_array = np.array(img)
            except Exception as pil_err:
                # PIL is the only option, no OpenCV fallback to avoid libGL dependency
                return {"error": f"Cannot open image file: {str(pil_err)}"}
            
            if img_array is None:
                return {"error": "Failed to load image array"}
            
            # Image properties
            height, width = img_array.shape[:2]

            # Calculate statistics (work across channels if present)
            mean_intensity = float(np.mean(img_array))
            std_intensity = float(np.std(img_array))
            contrast = float(std_intensity / (mean_intensity + 1e-6))
            
            # Determine image type
            if mean_intensity < 50:
                image_type = "Dark/Low Intensity Image"
                characteristics = "Low brightness, may indicate underexposed scan"
            elif mean_intensity > 200:
                image_type = "Bright/High Intensity Image"
                characteristics = "High brightness, may indicate overexposed scan"
            else:
                image_type = "Normal Intensity Image"
                characteristics = "Optimal brightness levels detected"
            
            # Quality assessment
            if contrast < 0.3:
                quality = "Low Contrast - May affect diagnosis"
            elif contrast > 1.5:
                quality = "High Contrast - Good for analysis"
            else:
                quality = "Moderate Contrast - Acceptable quality"
            
            # Generate findings
            findings = {
                "image_type": image_type,
                "dimensions": f"{width}x{height} pixels",
                # keep both programmatic numeric values and formatted strings
                "mean_intensity": f"{mean_intensity:.2f}",
                "mean_intensity_value": mean_intensity,
                "std_intensity_value": std_intensity,
                "contrast_ratio": f"{contrast:.2f}",
                "contrast_value": contrast,
                "characteristics": characteristics,
                "quality_assessment": quality,
                "recommendations": self._get_recommendations(mean_intensity, contrast)
            }
            
            return findings
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_recommendations(self, intensity, contrast):
        """Generate recommendations based on image analysis"""
        recommendations = []
        
        if intensity < 50:
            recommendations.append("Increase brightness/exposure for better visibility")
        elif intensity > 200:
            recommendations.append("Reduce brightness/exposure to prevent overexposure")
        
        if contrast < 0.3:
            recommendations.append("Improve contrast for better diagnostic accuracy")
        
        if not recommendations:
            recommendations.append("Image quality is suitable for analysis")
        
        return recommendations
    
    def ensemble_analysis(self, image_path):
        """Ensemble analysis with model predictions (simulated for lightweight version)"""
        # Get basic image analysis
        basic_analysis = self.analyze_image(image_path)
        
        if "error" in basic_analysis:
            return basic_analysis
        
        # Simulate ensemble predictions based on image characteristics
        mean_intensity = basic_analysis.get("mean_intensity_value", 128)
        contrast = basic_analysis.get("contrast_value", 0.5)
        
        # Simulate DenseNet121 prediction
        densenet_confidence = min(0.95, 0.5 + (contrast * 0.3))
        
        # Simulate MobileNetV2 prediction
        mobilenet_confidence = min(0.92, 0.45 + (contrast * 0.25))
        
        # Ensemble confidence (average)
        ensemble_confidence = (densenet_confidence + mobilenet_confidence) / 2
        
        # Determine risk level
        if ensemble_confidence > 0.8:
            risk_level = "High Risk"
            risk_color = "#ff4444"
        elif ensemble_confidence > 0.6:
            risk_level = "Moderate Risk"
            risk_color = "#ffaa44"
        else:
            risk_level = "Low Risk"
            risk_color = "#44ff44"
        
        return {
            "image_type": basic_analysis.get("image_type"),
            "dimensions": basic_analysis.get("dimensions"),
            "mean_intensity": basic_analysis.get("mean_intensity"),
            "contrast_ratio": basic_analysis.get("contrast_ratio"),
            "quality_assessment": basic_analysis.get("quality_assessment"),
            "recommendations": basic_analysis.get("recommendations"),
            "ensemble_confidence": ensemble_confidence,
            "densenet_confidence": densenet_confidence,
            "mobilenet_confidence": mobilenet_confidence,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "characteristics": basic_analysis.get("characteristics")
        }

def get_analyzer():
    """Get analyzer instance"""
    return MedicalImagingAnalyzer()


def extract_features_for_ml(image_path):
    """Helper that returns a simple feature dict suitable for ML training/prediction.

    Returns:
        dict: {"mean_intensity": float, "std_intensity": float, "contrast": float, "width": int, "height": int}
    """
    analyzer = get_analyzer()
    res = analyzer.analyze_image(image_path)
    if res.get("error"):
        raise RuntimeError(res["error"])

    return {
        "mean_intensity": float(res.get("mean_intensity_value", 0.0)),
        "std_intensity": float(res.get("std_intensity_value", 0.0)),
        "contrast": float(res.get("contrast_value", 0.0)),
        "width": int(res.get("dimensions", "0x0").split("x")[0]) if "dimensions" in res else 0,
        "height": int(res.get("dimensions", "0x0").split("x")[1].split()[0]) if "dimensions" in res else 0,
    }
