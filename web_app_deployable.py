"""
Deployable Flask API for Chest X-Ray Report Generation
This is a self-contained, independent Flask application for Render deployment.
"""

import os
import io
import re
import json
import html
import string
import textwrap
import unicodedata
import numpy as np
from PIL import Image
from datetime import datetime
from collections import OrderedDict
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET

# Optional spacy import
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Optional imports for enhanced features
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    GPT2_AVAILABLE = True
except ImportError:
    GPT2_AVAILABLE = False

try:
    from huggingface_hub import login as hf_login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
CHECKPOINT_FILE = './checkpoints/x_ray_model.pth.tar'
VOCAB_FILE = './checkpoints/vocab.json'  # Optional: pre-saved vocabulary
WEIGHTS_PATH = './weights/chexnet.pth.tar'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATURES_SIZE = 1024
EMBED_SIZE = 300
HIDDEN_SIZE = 256

basic_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ==================== TEXT UTILITIES ====================
def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))


def remove_non_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(text):
    return text.lower()


def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    return re.sub(r'\d+', '', text)


def normalize_text(text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    return text


# ==================== VOCABULARY CLASS ====================
class Vocabulary:
    """Vocabulary class that can be loaded from checkpoint or JSON file - NO DATASET NEEDED"""
    def __init__(self):
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}

    def __len__(self):
        return len(self.itos)
    
    def load_from_checkpoint(self, checkpoint):
        """Load vocabulary from model checkpoint if available"""
        if 'vocab_itos' in checkpoint and 'vocab_stoi' in checkpoint:
            self.itos = checkpoint['vocab_itos']
            self.stoi = checkpoint['vocab_stoi']
            print(f"✅ Vocabulary loaded from checkpoint: {len(self.itos)} words")
            return True
        return False
    
    def load_from_json(self, json_path):
        """Load vocabulary from JSON file"""
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.itos = {int(k): v for k, v in data['itos'].items()}
                self.stoi = data['stoi']
            print(f"✅ Vocabulary loaded from JSON: {len(self.itos)} words")
            return True
        return False
    
    def save_to_json(self, json_path):
        """Save vocabulary to JSON file for future use"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'itos': {str(k): v for k, v in self.itos.items()},
                'stoi': self.stoi
            }, f)
        print(f"✅ Vocabulary saved to {json_path}")


# ==================== MODEL CLASSES ====================
class DenseNet121(nn.Module):
    def __init__(self, out_size=14, checkpoint=None):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(weights='DEFAULT')
        num_classes = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_classes, out_size),
            nn.Sigmoid()
        )

        if checkpoint is not None and os.path.exists(checkpoint):
            try:
                ckpt = torch.load(checkpoint, map_location=torch.device('cpu'))
                state_dict = ckpt['state_dict']
                new_state_dict = OrderedDict()

                for k, v in state_dict.items():
                    if 'module' not in k:
                        k = f'module.{k}'
                    else:
                        k = k.replace('module.densenet121.features', 'features')
                        k = k.replace('module.densenet121.classifier', 'classifier')
                        k = k.replace('.norm.1', '.norm1')
                        k = k.replace('.conv.1', '.conv1')
                        k = k.replace('.norm.2', '.norm2')
                        k = k.replace('.conv.2', '.conv2')
                        new_state_dict[k] = v

                self.densenet121.load_state_dict(new_state_dict)
            except Exception as e:
                print(f"Could not load DenseNet checkpoint: {e}")

    def forward(self, x):
        return self.densenet121(x)


class EncoderCNN(nn.Module):
    def __init__(self, checkpoint=None):
        super(EncoderCNN, self).__init__()
        self.model = DenseNet121(checkpoint=checkpoint)
        for param in self.model.densenet121.parameters():
            param.requires_grad_(False)

    def forward(self, images):
        features = self.model.densenet121.features(images)
        batch, maps, size_1, size_2 = features.size()
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch, size_1 * size_2, maps)
        return features


class Attention(nn.Module):
    def __init__(self, features_size, hidden_size, output_size=1):
        super(Attention, self).__init__()
        self.W = nn.Linear(features_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, output_size)

    def forward(self, features, decoder_output):
        decoder_output = decoder_output.unsqueeze(1)
        w = self.W(features)
        u = self.U(decoder_output)
        scores = self.v(torch.tanh(w + u))
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * features, dim=1)
        weights = weights.squeeze(2)
        return context, weights


class DecoderRNN(nn.Module):
    def __init__(self, features_size, embed_size, hidden_size, vocab_size, device='cpu'):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + features_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(features_size, hidden_size)
        self.init_h = nn.Linear(features_size, hidden_size)
        self.init_c = nn.Linear(features_size, hidden_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        h, c = self.init_hidden(features)
        seq_len = len(captions[0]) - 1
        features_size = features.size(1)
        batch_size = captions.size(0)

        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(self.device)
        atten_weights = torch.zeros(batch_size, seq_len, features_size).to(self.device)

        for i in range(seq_len):
            context, attention = self.attention(features, h)
            inputs = torch.cat((embeddings[:, i, :], context), dim=1)
            h, c = self.lstm(inputs, (h, c))
            h = F.dropout(h, p=0.5)
            output = self.fc(h)
            outputs[:, i, :] = output
            atten_weights[:, i, :] = attention

        return outputs, atten_weights

    def init_hidden(self, features):
        features = torch.mean(features, dim=1)
        h = self.init_h(features)
        c = self.init_c(features)
        return h, c


class EncoderDecoderNet(nn.Module):
    def __init__(self, features_size, embed_size, hidden_size, vocabulary, encoder_checkpoint=None, device='cpu'):
        super(EncoderDecoderNet, self).__init__()
        self.vocabulary = vocabulary
        self.device = device
        self.encoder = EncoderCNN(checkpoint=encoder_checkpoint)
        self.decoder = DecoderRNN(
            features_size=features_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=len(self.vocabulary),
            device=device
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs, _ = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, max_length=25):
        caption = []
        with torch.no_grad():
            features = self.encoder(image)
            h, c = self.decoder.init_hidden(features)
            word = torch.tensor(self.vocabulary.stoi['<SOS>']).view(1, -1).to(self.device)
            embeddings = self.decoder.embedding(word).squeeze(0)

            for _ in range(max_length):
                context, _ = self.decoder.attention(features, h)
                inputs = torch.cat((embeddings, context), dim=1)
                h, c = self.decoder.lstm(inputs, (h, c))
                output = self.decoder.fc(F.dropout(h, p=0.5))
                output = output.view(1, -1)
                predicted = output.argmax(1)

                if self.vocabulary.itos[predicted.item()] == '<EOS>':
                    break

                caption.append(predicted.item())
                embeddings = self.decoder.embedding(predicted)

        return [self.vocabulary.itos[idx] for idx in caption]


# ==================== FLASK APP ====================
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for all routes
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global model variables - lazy loaded on demand
_models_cache = {}
_api_configured = False


def configure_apis():
    """Configure external APIs on startup (lightweight)"""
    global _api_configured
    if _api_configured:
        return
    
    # Configure Gemini API
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    if GOOGLE_API_KEY and GEMINI_AVAILABLE:
        genai.configure(api_key=GOOGLE_API_KEY)
        print("✅ Gemini API configured")
    else:
        print("⚠️ Gemini API not available")

    # Configure HuggingFace
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and HF_AVAILABLE:
        try:
            hf_login(token=hf_token, add_to_git_credential=False)
            print("✅ HuggingFace login successful")
        except Exception as e:
            print(f"⚠️ HuggingFace login failed: {e}")
    
    _api_configured = True


def get_xray_model():
    """Lazy load X-Ray model on first use"""
    if 'xray_model' in _models_cache:
        return _models_cache['xray_model'], _models_cache['xray_vocab']
    
    print("Loading X-Ray report generation model...")
    
    # Initialize vocabulary
    vocabulary = Vocabulary()
    vocab_loaded = vocabulary.load_from_json(VOCAB_FILE)
    
    # Load checkpoint
    checkpoint = None
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading checkpoint from {CHECKPOINT_FILE}")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
        if not vocab_loaded:
            vocab_loaded = vocabulary.load_from_checkpoint(checkpoint)
    
    if not vocab_loaded:
        print("⚠️ Could not load vocabulary")
    
    # Create model
    model = EncoderDecoderNet(
        features_size=FEATURES_SIZE,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocabulary=vocabulary,
        encoder_checkpoint=WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else None,
        device=DEVICE
    )
    model = model.to(DEVICE)
    
    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint['state_dict'])
            print("✅ X-Ray model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}")
    
    model.eval()
    
    # Convert to half precision (float16) to save memory
    if DEVICE == 'cuda':
        model = model.half()
    
    _models_cache['xray_model'] = model
    _models_cache['xray_vocab'] = vocabulary
    return model, vocabulary


def get_gpt2_model():
    """Lazy load GPT-2 model on first use"""
    if 'gpt2_model' in _models_cache:
        return _models_cache['gpt2_tokenizer'], _models_cache['gpt2_model']
    
    if not GPT2_AVAILABLE:
        return None, None
    
    try:
        print("Loading GPT-2 model...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.eval()
        
        # Move to CPU to save GPU memory if not available
        if DEVICE == 'cpu':
            model = model.to(DEVICE)
        
        print("✅ GPT-2 loaded successfully")
        _models_cache['gpt2_tokenizer'] = tokenizer
        _models_cache['gpt2_model'] = model
        return tokenizer, model
    except Exception as e:
        print(f"⚠️ GPT-2 loading failed: {e}")
        return None, None


def initialize_models():
    """Lightweight initialization - just configure APIs"""
    configure_apis()
    print("✅ API configuration complete (Models loaded on-demand)")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(file):
    """Process uploaded image for model inference"""
    image_bytes = io.BytesIO(file.read())
    image = np.array(Image.open(image_bytes).convert("L"))
    image = np.expand_dims(image, axis=-1)
    image = image.repeat(3, axis=-1)
    image = basic_transforms(image=image)["image"]
    image = image.to(DEVICE)
    return image


def get_detailed_report_gpt2(caption, max_new_tokens=150):
    """Generate detailed report using GPT-2 (lazy loaded)"""
    gpt2_tokenizer, gpt2_model = get_gpt2_model()
    
    if gpt2_tokenizer is None or gpt2_model is None:
        return "Detailed report generation not available."

    try:
        prompt = (
            "Given the following short AI-generated caption of a chest X-ray, write a detailed, "
            "professional, patient-friendly radiology report with explanations, findings, and recommendations.\n"
            f"Caption: {caption}\n"
            "Explanation:"
        )
        input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = gpt2_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                pad_token_id=gpt2_tokenizer.eos_token_id,
            )
        generated_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        detailed_report = generated_text.split("Explanation:")[-1].strip()
        if not detailed_report:
            detailed_report = "Explanation could not be generated at this time."
        return detailed_report
    except Exception as e:
        print(f"Error generating GPT-2 report: {e}")
        return "Detailed report generation not available."


def extract_clinical_terms(caption, max_terms=5):
    """Extract clinical terms from caption"""
    clinical_terms_list = [
        "granuloma", "consolidation", "effusion", "nodule", "atelectasis",
        "infiltrate", "fibrosis", "opacity", "pneumothorax", "cardiomegaly",
        "edema", "calcification", "pleural thickening", "mass", "emphysema",
        "pneumonia", "sarcoidosis", "hyperinflation", "collapse", "lesion",
        "pleural effusion", "interstitial markings", "hilar enlargement",
        "lymphadenopathy", "bronchiectasis", "cavity", "scar", "infection",
        "pleural fluid", "pleural plaque", "pleural calcification", "reticulation",
        "honeycombing", "volume loss", "air trapping", "bullae", "consolidations",
        "nodules", "masses", "normal", "clear", "unremarkable"
    ]
    caption_lower = caption.lower()
    found_terms = []
    for term in clinical_terms_list:
        if term in caption_lower and term not in found_terms:
            found_terms.append(term)
        if len(found_terms) == max_terms:
            break
    return ", ".join(found_terms) if found_terms else "No specific clinical terms identified."


def generate_enhanced_report(text):
    """Generate enhanced report using Gemini API"""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    if not GOOGLE_API_KEY or not GEMINI_AVAILABLE:
        return generate_fallback_report(text)

    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
            Given the following radiology report summary (1-2 sentences), generate a structured report in the following format.

            CLINICAL TAGS
            <comma-separated main findings or conditions, both normal and abnormal>

            KEY FINDINGS
            <one or two clear, concise summary sentences covering key findings>

            ABSTRACT
            COMPARISON: <Describe comparison if present, otherwise state "Not applicable" or "None.">
            INDICATION: <Describe indication/reason for the exam, or state "None provided." if absent.>
            FINDINGS: <Summarize findings in one or two sentences, or state "None reported." if absent.>
            IMPRESSION: <Summarize impression using all clinical information, rephrased as a single concise paragraph.>

            Original report: {text}
        """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return generate_fallback_report(text)


def generate_fallback_report(text):
    """Generate fallback report when Gemini is not available"""
    return f"""CLINICAL TAGS
{text}

KEY FINDINGS
{text}

ABSTRACT
COMPARISON: None available.
INDICATION: Chest X-ray analysis.
FINDINGS: {text}
IMPRESSION: AI-generated analysis based on chest X-ray findings. Please consult a medical professional for accurate diagnosis.
"""


def verify_xray_image(image_file):
    """Verify if the uploaded image is a chest X-ray using Gemini Vision"""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    if not GOOGLE_API_KEY or not GEMINI_AVAILABLE:
        return True, "Verification skipped (API not configured)"

    try:
        image_file.stream.seek(0)
        image_bytes = image_file.read()
        image_file.stream.seek(0)

        pil_image = Image.open(io.BytesIO(image_bytes))
        model_vision = genai.GenerativeModel('gemini-2.0-flash')

        image_parts = [{
            "mime_type": f"image/{pil_image.format.lower() if pil_image.format else 'jpeg'}",
            "data": image_bytes
        }]

        prompt = """
        Analyze this image and determine if it is a medical chest X-ray (radiograph).
        Respond in JSON format:
        {"is_chest_xray": true/false, "confidence": "high/medium/low", "reason": "brief explanation"}
        """

        response = model_vision.generate_content([prompt, image_parts[0]])
        response_text = response.text.strip()

        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            is_valid = result.get("is_chest_xray", False)
            reason = result.get("reason", "Unknown")
            return is_valid, reason if not is_valid else "Valid chest X-ray"

        return True, "Could not parse verification response"
    except Exception as e:
        print(f"Verification error: {e}")
        return True, f"Verification skipped due to error: {str(e)}"


# ==================== API ROUTES ====================

@app.route("/", methods=["GET"])
def serve_index():
    """Serve the main UI"""
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, images, etc.)"""
    if os.path.exists(os.path.join('static', path)):
        return send_from_directory('static', path)
    # If file not found, serve index.html (for SPA routing)
    return send_from_directory('static', 'index.html')


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Chest X-Ray Report Generator API",
        "version": "2.0.0",
        "note": "No dataset dependency - fully deployable",
        "endpoints": {
            "/": "UI (GET)",
            "/api/generate": "Generate report from X-ray image (POST)",
            "/api/verify": "Verify if image is a chest X-ray (POST)",
            "/api/download-pdf": "Download report as PDF (POST)",
            "/api/health": "Health check (GET)"
        }
    })


@app.route("/api/generate", methods=["POST"])
def generate_report():
    """
    Generate a diagnostic report from a chest X-ray image
    
    Request: multipart/form-data with 'image' file
    Optional fields: 'name', 'age', 'verify' (boolean)
    """
    # Lazy load model on first request
    xray_model, vocabulary = get_xray_model()
    if xray_model is None:
        return jsonify({"error": "Model not initialized"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400

    # Optional verification
    verify = request.form.get("verify", "false").lower() == "true"
    if verify:
        is_valid, verification_message = verify_xray_image(file)
        if not is_valid:
            return jsonify({
                "error": "Invalid image",
                "verification_message": verification_message
            }), 400

    try:
        # Process image and generate caption
        image_tensor = process_image(file)
        with torch.no_grad():
            caption_words = xray_model.generate_caption(image_tensor.unsqueeze(0), max_length=25)
        caption = " ".join(caption_words)
        
        # Clear image tensor to free memory
        del image_tensor
        torch.cuda.empty_cache()

        # Generate detailed reports
        detailed_report_gpt2 = get_detailed_report_gpt2(caption)
        clinical_terms = extract_clinical_terms(caption)
        enhanced_report = generate_enhanced_report(caption)

        # Get metadata
        name = request.form.get("name", "")
        age = request.form.get("age", "")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            "success": True,
            "timestamp": timestamp,
            "patient_info": {"name": name, "age": age},
            "caption": caption,
            "clinical_terms": clinical_terms,
            "detailed_report_gpt2": detailed_report_gpt2,
            "enhanced_report": enhanced_report,
            "similar_reports": []  # Removed - no dataset dependency
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Clear cache on error
        torch.cuda.empty_cache()
        return jsonify({"error": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
def verify_image():
    """
    Verify if an uploaded image is a chest X-ray
    
    Request: multipart/form-data with 'image' file
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    is_valid, message = verify_xray_image(file)
    return jsonify({
        "is_valid": is_valid,
        "message": message
    })


@app.route("/api/download-pdf", methods=["POST"])
def download_pdf():
    """
    Generate and download a PDF report
    
    Request JSON: {
        "caption": "...",
        "enhanced_report": "...",
        "clinical_terms": "...",
        "name": "...",
        "age": "..."
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Report data required"}), 400

    caption = data.get("caption", "No caption provided")
    enhanced_report = data.get("enhanced_report", "")
    clinical_terms = data.get("clinical_terms", "")
    name = data.get("name", "")
    age = data.get("age", "")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    left_margin = 50
    top_margin = height - 50
    line_gap = 18

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_margin, top_margin, "AI Chest X-Ray Diagnostic Report")

    c.setFont("Helvetica", 10)
    c.drawString(left_margin, top_margin - line_gap, f"Generated on: {timestamp}")
    c.drawString(left_margin, top_margin - 2 * line_gap, f"Name: {name}    Age: {age}")

    y = top_margin - 4 * line_gap

    # AI Caption
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "AI Caption:")
    y -= line_gap
    c.setFont("Helvetica", 12)
    for line in textwrap.wrap(caption, 70):
        c.drawString(left_margin, y, line)
        y -= line_gap

    # Clinical Terms
    y -= line_gap // 2
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Clinical Terms:")
    y -= line_gap
    c.setFont("Helvetica", 12)
    for line in textwrap.wrap(clinical_terms, 70):
        c.drawString(left_margin, y, line)
        y -= line_gap

    # Enhanced Report
    y -= line_gap // 2
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Detailed Report:")
    y -= line_gap
    c.setFont("Helvetica", 10)
    for line in textwrap.wrap(enhanced_report, 80):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(left_margin, y, line)
        y -= 14

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Xray_Report.pdf",
        mimetype="application/pdf"
    )


# ==================== MAIN ====================
# Initialize models on import (for Gunicorn)
with app.app_context():
    initialize_models()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
