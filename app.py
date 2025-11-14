import os
import base64
import logging
import time
from flask import Flask, render_template, request, Response, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
try:
    from groq import Groq
except Exception:
    Groq = None
import markdown
from markupsafe import Markup

# Lazy loading - don't import heavy models at startup
import joblib
from pathlib import Path
from PIL import Image
import numpy as np

# These will be imported on first use
get_analyzer = None
extract_features_for_ml = None
tf = None

# Lazy loading function for models
def load_models():
    """Lazy load models on first use"""
    global get_analyzer, extract_features_for_ml, tf
    if get_analyzer is None:
        try:
            from ml_model_lite import get_analyzer as lite_analyzer, extract_features_for_ml as lite_features
            get_analyzer = lite_analyzer
            extract_features_for_ml = lite_features
            logging.info("✓ Loaded lightweight ML model")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise

# Load environment variables from .env file
load_dotenv()
logging.info("Environment variables loaded from .env file.") # <--- ADDED THIS LINE

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dicom"}

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REACT_BUILD_DIR = os.path.join(BASE_DIR, "frontend", "build")

app = Flask(__name__, static_folder=os.path.join(REACT_BUILD_DIR, "static"), static_url_path="/static")
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Allow a GROQ API key to be provided via environment variable to avoid
# having to type it into the UI every time. This is useful for local testing.
ENV_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if ENV_GROQ_API_KEY:
    ENV_GROQ_API_KEY = ENV_GROQ_API_KEY.strip()
logging.info(f"GROQ_API_KEY loaded: {bool(ENV_GROQ_API_KEY)}") # <--- ADDED THIS LINE

# Load your trained CT/MRI classifier
TRAINED_CT_MRI_MODEL = None

def load_trained_ct_mri_model():
    """Load your trained CT/MRI classifier model"""
    global TRAINED_CT_MRI_MODEL
    try:
        import tensorflow as tf
        model_path = Path("models/ct_mri_classifier.h5")
        if model_path.exists():
            try:
                # Try to load with custom_objects if needed
                TRAINED_CT_MRI_MODEL = tf.keras.models.load_model(str(model_path))
                logging.info(f"✓ Trained CT/MRI Classifier loaded from {model_path}")
                return TRAINED_CT_MRI_MODEL
            except Exception as load_err:
                logging.warning(f"Could not load model with default settings: {load_err}")
                logging.warning("Model loading will be deferred to first use")
                return None
        else:
            logging.warning(f"Trained CT/MRI model not found at {model_path}")
            return None
    except Exception as e:
        logging.warning(f"Error loading trained CT/MRI model (will retry on first use): {e}")
        return None

def predict_ct_mri_classification(image_path):
    """Use your trained CT/MRI classifier to predict"""
    try:
        if TRAINED_CT_MRI_MODEL is None:
            return None
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = TRAINED_CT_MRI_MODEL.predict(img_array, verbose=0)[0][0]
        
        # Interpret prediction
        class_idx = 1 if prediction > 0.5 else 0
        class_name = "MRI" if class_idx == 1 else "CT"
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        
        return {
            "predicted_class": class_name,
            "confidence": float(confidence),
            "raw_prediction": float(prediction)
        }
    except Exception as e:
        logging.error(f"Error in CT/MRI prediction: {e}")
        return None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Your medical analysis prompt
MEDICAL_QUERY = """
You are a medical imaging expert. Analyze this medical image and provide a brief, concise summary in 2-3 paragraphs:

1. What type of image is this and what body part does it show?
2. What are the main findings and any abnormalities detected?
3. What are the likely diseases or conditions based on these findings?

IMPORTANT: Format disease names in **bold** (e.g., **Pneumonia**, **Fracture**) to make them stand out.

Keep the explanation simple and direct. Avoid lengthy details and technical jargon. Focus only on the key observations and most likely diagnoses.
"""

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """Analyze image using Groq API or fallback to ML models"""
    # Lazy load models on first use
    load_models()
    
    start_time = time.time()
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Allowed image types are png, jpg, jpeg, dicom"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        logging.info(f"Image saved in {time.time() - start_time:.2f}s")

        # Use local ML models (Groq API disabled for reliability)
        ml_start = time.time()
        result = ml_analyze_image()
        logging.info(f"ML analysis completed in {time.time() - ml_start:.2f}s, Total time: {time.time() - start_time:.2f}s")
        return result
    
    except Exception as e:
        logging.exception("Error while performing analysis")
        return jsonify({"error": f"Error during analysis: {str(e)}"}), 500


@app.route("/api/ml-analyze", methods=["POST"])
def ml_analyze_image():
    """Analyze image using Deep Learning models (DenseNet + ResNet)"""
    # Lazy load models on first use
    load_models()
    
    start_time = time.time()
    try:
        logging.info(f"Request files: {request.files.keys()}")
        logging.info(f"Request form: {request.form.keys()}")
        logging.info(f"Content-Type: {request.content_type}")
        
        if "image" not in request.files:
            logging.error("No 'image' field in request.files")
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        logging.info(f"File object: {file}, filename: {file.filename}")
        logging.info(f"File stream position before read: {file.stream.tell()}")
        
        if file.filename == "":
            logging.error("File filename is empty")
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            logging.error(f"File type not allowed: {file.filename}")
            return jsonify({"error": "Allowed image types are png, jpg, jpeg, dicom"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        # Read file content and save explicitly
        try:
            # Reset stream to beginning
            file.stream.seek(0)
            file_content = file.read()
            logging.info(f"File content read: {len(file_content)} bytes")
            if not file_content:
                logging.error(f"File content is empty: {file.filename}")
                return jsonify({"error": "Uploaded file is empty"}), 400
            
            with open(filepath, 'wb') as f:
                f.write(file_content)
            
            file_size = len(file_content)
            logging.info(f"File saved: {filepath}, size: {file_size} bytes")
        except Exception as save_err:
            logging.error(f"Error saving file: {save_err}")
            return jsonify({"error": f"Error saving file: {str(save_err)}"}), 500

        # Get analyzer instance
        analyzer = get_analyzer()

        # Use full analyzer for disease detection if available
        logging.info(f"Starting image analysis for: {filepath}")
        analysis_start = time.time()
        
        # Verify file exists and has content
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return jsonify({"error": "File not found after upload"}), 400
        
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            logging.error(f"File is empty after save: {filepath}")
            return jsonify({"error": "Uploaded file is empty"}), 400
        
        logging.info(f"File verified: {file_size} bytes")
        
        # PRIORITY: Use your trained CT/MRI classifier first
        ct_mri_result = predict_ct_mri_classification(filepath)
        if ct_mri_result:
            logging.info(f"✓ Using TRAINED CT/MRI Classifier: {ct_mri_result['predicted_class']} ({ct_mri_result['confidence']*100:.1f}%)")
            confidence_percent = ct_mri_result['confidence'] * 100
            confidence_color = '#28a745' if confidence_percent > 90 else '#ffa500' if confidence_percent > 70 else '#ff6b6b'
            
            # Also get disease detection from medical analyzer
            try:
                analyzer = get_analyzer()
                if hasattr(analyzer, 'ensemble_analysis'):
                    findings = analyzer.ensemble_analysis(filepath)
                    disease_info = format_ml_analysis(findings) if findings else ""
                else:
                    disease_info = ""
            except Exception as e:
                logging.warning(f"Could not get disease analysis: {e}")
                disease_info = ""
            
            html_result = f"""
            <div style='font-family: "Segoe UI", sans-serif; width: 100%;'>
                <div style='background: linear-gradient(135deg, rgba(100, 200, 255, 0.15) 0%, rgba(100, 200, 255, 0.05) 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #64c8ff; margin-bottom: 15px;'>
                    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 15px;'>
                        <span style='font-size: 24px;'>🎯</span>
                        <h3 style='margin: 0; color: #64c8ff; font-size: 18px; font-weight: 600;'>TRAINED CT/MRI CLASSIFICATION</h3>
                    </div>
                    
                    <div style='background: rgba(100, 200, 255, 0.08); padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                        <p style='margin: 12px 0 0 0; color: #b0b0b0; font-size: 12px; font-weight: 500; text-transform: uppercase;'>Imaging Type</p>
                        <p style='margin: 6px 0 0 0; color: #64c8ff; font-size: 24px; font-weight: 700;'>{ct_mri_result['predicted_class']}</p>
                    </div>
                    
                    <div style='background: rgba(100, 200, 255, 0.08); padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                        <p style='margin: 0 0 8px 0; color: #b0b0b0; font-size: 12px; font-weight: 500; text-transform: uppercase;'>Classification Confidence</p>
                        <div style='background: rgba(100, 200, 255, 0.1); height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 8px;'>
                            <div style='background: linear-gradient(90deg, {confidence_color} 0%, {confidence_color}dd 100%); height: 100%; width: {min(int(confidence_percent * 2.5), 250)}px; border-radius: 4px;'></div>
                        </div>
                        <p style='margin: 0; color: {confidence_color}; font-size: 18px; font-weight: 700;'>{confidence_percent:.2f}%</p>
                    </div>
                    
                    <div style='color: #808080; font-size: 12px; border-top: 1px solid rgba(100, 200, 255, 0.1); padding-top: 12px;'>
                        <span style='color: #64c8ff;'>✓</span> Specialized CT/MRI Classifier (99.66% accuracy)
                    </div>
                </div>
                
                {disease_info}
            </div>
            """
            return jsonify({"result": html_result, "analysis": ct_mri_result, "source": "TRAINED_CT_MRI_MODEL"}), 200
        
        # Fallback to full analyzer for disease detection if available
        logging.info("Trained CT/MRI model result unavailable, falling back to full medical analyzer...")
        
        # Use ensemble analysis if available (full model with disease detection)
        if hasattr(analyzer, 'ensemble_analysis'):
            logging.info("Using ensemble analysis for disease detection...")
            findings = analyzer.ensemble_analysis(filepath)
            logging.info(f"Ensemble findings keys: {findings.keys()}")
            # Use format_ml_analysis for full model results
            html_result = format_ml_analysis(findings)
            logging.info(f"HTML result length: {len(html_result)}")
        else:
            # Fallback to lightweight analyzer
            logging.info("Using lightweight analyzer...")
            findings = analyzer.analyze_image(filepath)
            
            # Format lightweight analyzer results
            html_result = "<div style='font-family: Arial, sans-serif;'>"
            if 'error' in findings:
                html_result += f"<p style='color:red;'>Error: {findings['error']}</p>"
            else:
                html_result += f"<h3>Image Quality Analysis</h3><p><strong>Type:</strong> {findings.get('image_type')}<br/>"
                html_result += f"<strong>Dimensions:</strong> {findings.get('dimensions')}<br/>"
                html_result += f"<strong>Mean intensity:</strong> {findings.get('mean_intensity')}<br/>"
                html_result += f"<strong>Contrast:</strong> {findings.get('contrast_ratio')}<br/>"
                html_result += f"<strong>Quality:</strong> {findings.get('quality_assessment')}<br/></p>"
                html_result += "<h4>Recommendations</h4><ul>"
                for r in findings.get('recommendations', []):
                    html_result += f"<li>{r}</li>"
                html_result += "</ul>"
            html_result += "</div>"
        
        logging.info(f"Image analysis completed in {time.time() - analysis_start:.2f}s")
        logging.info(f"Final HTML result: {html_result[:200]}...")

        return jsonify({"result": html_result, "analysis": findings}), 200
    
    except Exception as e:
        logging.exception("Error during ML analysis")
        return jsonify({"error": f"Error during analysis: {str(e)}"}), 500


def format_ml_analysis(analysis_result):
    """Format ML analysis results as HTML using trained medical models"""
    try:
        if "error" in analysis_result:
            return f"<p style='color: red;'><strong>Error:</strong> {analysis_result['error']}</p>"
        
        html = """
        <style>
            .ml-card { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .dataset-box { 
                background: linear-gradient(135deg, rgba(100, 200, 255, 0.15) 0%, rgba(100, 200, 255, 0.05) 100%);
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 20px; 
                border-left: 5px solid #64c8ff;
                box-shadow: 0 4px 15px rgba(100, 200, 255, 0.1);
            }
            .ensemble-box {
                background: linear-gradient(135deg, rgba(100, 200, 255, 0.15) 0%, rgba(100, 200, 255, 0.05) 100%);
                padding: 18px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 5px solid #64c8ff;
                box-shadow: 0 4px 20px rgba(100, 200, 255, 0.15);
            }
            .model-box {
                background: linear-gradient(135deg, rgba(100, 150, 255, 0.12) 0%, rgba(100, 150, 255, 0.02) 100%);
                padding: 16px;
                margin-bottom: 16px;
                border-radius: 10px;
                border-left: 5px solid #6496ff;
                box-shadow: 0 4px 15px rgba(100, 150, 255, 0.1);
            }
            .confidence-bar {
                display: inline-block;
                height: 8px;
                background: linear-gradient(90deg, #64c8ff 0%, #9664ff 100%);
                border-radius: 4px;
                margin-left: 8px;
            }
            .finding-item {
                padding: 10px;
                margin: 8px 0;
                background: rgba(100, 200, 255, 0.08);
                border-radius: 6px;
                border-left: 3px solid #64c8ff;
            }
        </style>
        <div class='ml-card' style='background: transparent; padding: 0;'>
        """
        
        # Trained Datasets Info
        if "trained_datasets" in analysis_result:
            html += "<div class='dataset-box'>"
            html += "<p style='margin: 0 0 12px 0; font-size: 13px; color: #64c8ff; font-weight: 600;'>📚 TRAINED ON MEDICAL DATASETS</p>"
            for dataset in analysis_result["trained_datasets"]:
                html += f"<p style='margin: 6px 0; font-size: 12px; color: #b0b0b0; display: flex; align-items: center;'><span style='color: #64c8ff; margin-right: 8px;'>✓</span>{dataset}</p>"
            html += "</div>"
        
        # Ensemble Results
        if "ensemble_confidence" in analysis_result:
            confidence = analysis_result['ensemble_confidence']
            color = "#ff6b6b" if confidence >= 85 else "#ffa500" if confidence >= 70 else "#ffd700" if confidence >= 50 else "#64ff96"
            confidence_level = "🔴 High Risk" if confidence >= 85 else "🟠 Moderate-High" if confidence >= 70 else "🟡 Moderate" if confidence >= 50 else "🟢 Low Risk"
            confidence_width = min(int(confidence * 2.5), 250)
            html += f"""
            <div class='ensemble-box'>
                <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;'>
                    <h3 style='margin: 0; color: #64c8ff; font-size: 16px; font-weight: 600;'>🤖 ML ENSEMBLE ANALYSIS</h3>
                    <span style='background: {color}; color: white; padding: 8px 16px; border-radius: 25px; font-size: 13px; font-weight: 700; display: inline-block; white-space: nowrap; box-shadow: 0 4px 12px rgba(0,0,0,0.3);'>{confidence_level}</span>
                </div>
                <div style='margin-bottom: 15px;'>
                    <p style='margin: 0 0 8px 0; color: #b0b0b0; font-size: 12px; font-weight: 500;'>CONFIDENCE SCORE</p>
                    <div style='background: rgba(100, 200, 255, 0.1); height: 12px; border-radius: 6px; overflow: hidden; border: 1px solid rgba(100, 200, 255, 0.2);'>
                        <div style='background: linear-gradient(90deg, {color} 0%, {color}dd 100%); height: 100%; width: {confidence_width}px; border-radius: 6px;'></div>
                    </div>
                    <p style='margin: 8px 0 0 0; color: {color}; font-size: 18px; font-weight: 700;'>{confidence:.1f}%</p>
                </div>
                <div style='background: rgba(100, 200, 255, 0.08); padding: 12px; border-radius: 6px; margin-bottom: 12px;'>
                    <p style='margin: 0; color: #b0b0b0; font-size: 12px;'><strong>Clinical Assessment:</strong></p>
                    <p style='margin: 6px 0 0 0; color: #64c8ff; font-size: 13px;'>{analysis_result['recommendation']}</p>
                </div>
                <p style='margin: 0; font-size: 11px; color: #808080; border-top: 1px solid rgba(100, 200, 255, 0.1); padding-top: 10px;'>
                    <span style='color: #ffa500;'>⚠️</span> ML predictions should be reviewed by a radiologist for clinical confirmation.
                </p>
            </div>
            """
        
        # DenseNet Results (CheXpert-trained)
        if "densenet_result" in analysis_result and "error" not in analysis_result["densenet_result"]:
            dn = analysis_result["densenet_result"]
            html += f"""
            <div class='model-box'>
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 12px;'>
                    <span style='font-size: 20px;'>🔬</span>
                    <div>
                        <h4 style='margin: 0; color: #64c8ff; font-size: 14px; font-weight: 600;'>DENSENET121 ANALYSIS</h4>
                        <p style='margin: 4px 0 0 0; font-size: 11px; color: #808080;'>CheXpert-trained • 224,316 chest X-rays</p>
                    </div>
                </div>
                <div style='background: rgba(100, 200, 255, 0.08); padding: 12px; border-radius: 6px; margin-bottom: 12px;'>
                    <p style='margin: 0 0 6px 0; color: #b0b0b0; font-size: 11px; font-weight: 500;'>PRIMARY FINDING</p>
                    <p style='margin: 0; color: #ff6b6b; font-size: 15px; font-weight: 700;'>{dn['top_prediction']} <span style='color: #64c8ff; font-size: 13px;'>({dn['confidence']:.1f}%)</span></p>
                </div>
                <p style='margin: 0 0 10px 0; color: #b0b0b0; font-size: 12px; font-weight: 500;'>DIFFERENTIAL DIAGNOSIS</p>
            """
            for i, pred in enumerate(dn["predictions"][:5], 1):
                bar_width = int(pred['confidence'] * 1.5)
                html += f"""
                <div class='finding-item'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'>
                        <span style='color: #b0b0b0; font-size: 12px;'><strong>{i}.</strong> <span style='color: #64c8ff;'>{pred['class']}</span></span>
                        <span style='color: #64c8ff; font-weight: 600; font-size: 12px;'>{pred['confidence']:.1f}%</span>
                    </div>
                    <div style='background: rgba(100, 200, 255, 0.1); height: 6px; border-radius: 3px; overflow: hidden;'>
                        <div style='background: linear-gradient(90deg, #64c8ff 0%, #9664ff 100%); height: 100%; width: {bar_width}px;'></div>
                    </div>
                </div>
                """
            html += "</div>"
        
        # MobileNetV2 Results (MIMIC-CXR-trained)
        if "mobilenet_result" in analysis_result and "error" not in analysis_result["mobilenet_result"]:
            mn = analysis_result["mobilenet_result"]
            html += f"""
            <div class='model-box' style='border-left-color: #64ff96; background: linear-gradient(135deg, rgba(100, 255, 150, 0.12) 0%, rgba(100, 255, 150, 0.02) 100%);'>
                <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 12px;'>
                    <span style='font-size: 20px;'>🔬</span>
                    <div>
                        <h4 style='margin: 0; color: #64ff96; font-size: 14px; font-weight: 600;'>MOBILENETV2 ANALYSIS</h4>
                        <p style='margin: 4px 0 0 0; font-size: 11px; color: #808080;'>MIMIC-CXR-trained • 377,110 chest X-rays</p>
                    </div>
                </div>
                <div style='background: rgba(100, 255, 150, 0.08); padding: 12px; border-radius: 6px; margin-bottom: 12px;'>
                    <p style='margin: 0 0 6px 0; color: #b0b0b0; font-size: 11px; font-weight: 500;'>PRIMARY FINDING</p>
                    <p style='margin: 0; color: #64ff96; font-size: 15px; font-weight: 700;'>{mn['top_prediction']} <span style='color: #64c8ff; font-size: 13px;'>({mn['confidence']:.1f}%)</span></p>
                </div>
                <p style='margin: 0 0 10px 0; color: #b0b0b0; font-size: 12px; font-weight: 500;'>DIFFERENTIAL DIAGNOSIS</p>
            """
            for i, pred in enumerate(mn["predictions"][:5], 1):
                bar_width = int(pred['confidence'] * 1.5)
                html += f"""
                <div class='finding-item' style='border-left-color: #64ff96;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'>
                        <span style='color: #b0b0b0; font-size: 12px;'><strong>{i}.</strong> <span style='color: #64ff96;'>{pred['class']}</span></span>
                        <span style='color: #64ff96; font-weight: 600; font-size: 12px;'>{pred['confidence']:.1f}%</span>
                    </div>
                    <div style='background: rgba(100, 255, 150, 0.1); height: 6px; border-radius: 3px; overflow: hidden;'>
                        <div style='background: linear-gradient(90deg, #64ff96 0%, #64c8ff 100%); height: 100%; width: {bar_width}px;'></div>
                    </div>
                </div>
                """
            html += "</div>"
        
        # Legacy ResNet support
        if "resnet_result" in analysis_result and "error" not in analysis_result["resnet_result"]:
            rn = analysis_result["resnet_result"]
            html += f"""
            <div style='background: white; padding: 12px; margin-bottom: 12px; border-radius: 5px; border: 1px solid #ddd;'>
            <h4 style='margin-top: 0;'>📊 ResNet50 Analysis</h4>
            <p style='margin: 5px 0;'><strong>Top Prediction:</strong> {rn['top_prediction']}</p>
            <p style='margin: 5px 0;'><strong>Confidence:</strong> {rn['confidence']:.1f}%</p>
            <ul style='margin: 5px 0; padding-left: 20px;'>
            """
            for pred in rn["predictions"][:3]:
                html += f"<li>{pred['class']}: {pred['confidence']:.1f}%</li>"
            html += "</ul></div>"
        
        html += "</div>"
        return html
    except Exception as e:
        return f"<p style='color: red;'>Error formatting results: {str(e)}</p>"


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(REACT_BUILD_DIR, path)):
        return send_from_directory(REACT_BUILD_DIR, path)
    return send_from_directory(REACT_BUILD_DIR, "index.html")


@app.route('/favicon.ico')
def favicon():
    return Response(status=204)

# Skip pre-loading to avoid startup issues
# Models will be loaded on first use (lazy loading)
@app.before_request
def preload_models():
    """Skip pre-loading - models load on first use"""
    if not hasattr(app, 'models_loaded'):
        logging.info("App started - models will load on first use (lazy loading)")
        app.models_loaded = True

@app.route("/api/ask-medical", methods=["POST"])
def ask_medical():
    """Medical Q&A endpoint using GROQ AI"""
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid or missing JSON body",
                "success": False
            }), 400

        user_question = (data.get("question") or "").strip()
        disease_context = (data.get("disease") or "").strip()  # Optional: disease from prediction
        
        if not user_question:
            return jsonify({"error": "No question provided"}), 400
        
        # Check if GROQ is available
        if Groq is None:
            return jsonify({
                "error": "Medical Q&A service is not available. GROQ library not loaded.",
                "available": False
            }), 503

        if not ENV_GROQ_API_KEY or not isinstance(ENV_GROQ_API_KEY, str):
            logging.error(f"GROQ API key is invalid or not set. Current value: '{ENV_GROQ_API_KEY}'") # <--- ADDED THIS LINE
            return jsonify({
                "error": "Medical Q&A service is not available. Please set up a valid GROQ API key in .env file.",
                "available": False
            }), 503

        try:
            # Initialize GROQ client with minimal parameters
            try:
                client = Groq(api_key=ENV_GROQ_API_KEY)
            except TypeError as te:
                # Handle version compatibility issues
                logging.warning(f"Groq client init error: {te}, retrying with alternative method")
                import os
                os.environ['GROQ_API_KEY'] = ENV_GROQ_API_KEY
                client = Groq()

            # Build system message with medical context
            system_message = """You are a highly knowledgeable medical AI assistant with expertise in healthcare and disease management.
            
IMPORTANT GUIDELINES:
1. Provide accurate, evidence-based medical information
2. Use professional medical terminology where appropriate, but explain in simple terms
3. Always include a DISCLAIMER: "This information is for educational purposes only. Please consult a qualified healthcare professional for diagnosis and treatment."
4. Structure answers clearly with headings and bullet points
5. Be empathetic and patient in your responses
6. Avoid diagnosing conditions - focus on information and education
7. If asked about symptoms, suggest seeing a healthcare provider
8. Provide prevention and lifestyle tips when relevant
9. Be concise but comprehensive
10. Format responses in clean Markdown for readability"""
            
            # Add disease context if provided
            user_message = user_question
            if disease_context:
                user_message = f"Context: The user's medical image showed potential signs of {disease_context}.\n\nTheir question: {user_question}"
            
            # Call GROQ API using chat completions
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Current production model
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )

            # Extract response
            answer = completion.choices[0].message.content
            
            # Add disclaimer if not already present
            if "educational purposes" not in answer.lower():
                disclaimer = "\n\n⚠️ **DISCLAIMER**: This information is for educational purposes only. Please consult a qualified healthcare professional for diagnosis and treatment."
                answer = answer + disclaimer
            
            logging.info(f"Medical Q&A responded in {time.time():.2f}s")
            
            return jsonify({
                "answer": answer,
                "success": True,
                "disease_context": disease_context if disease_context else None
            }), 200
            
        except Exception as groq_error:
            logging.error(f"GROQ API error: {str(groq_error)}")
            return jsonify({
                "error": f"Error processing your question: {str(groq_error)}",
                "success": False
            }), 500
    
    except Exception as e:
        logging.error(f"Medical Q&A endpoint error: {str(e)}")
        return jsonify({
            "error": f"Server error: {str(e)}",
            "success": False
        }), 500

if __name__ == "__main__":
    logging.info("Starting Flask app...")
    app.run(debug=True)
