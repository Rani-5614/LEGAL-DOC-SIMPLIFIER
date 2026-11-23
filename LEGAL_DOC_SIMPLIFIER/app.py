# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timezone, timedelta
import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import tempfile
from gtts import gTTS
import PyPDF2
import docx
from PIL import Image
import pytesseract
import pdf2image
import easyocr
import cv2
import numpy as np
from googletrans import Translator
import time
import threading
import math
import requests
import pickle
from flask import send_file, make_response
from io import BytesIO
import io
# Import our ML model
from models.simplifier_model import LegalTextSimplifier
from models.legal_analyzer import (detect_document_type, analyze_risks, extract_deadlines,
                                   get_specialization_profile, days_until)
from models import lawyer_finder
from datetime import date
import google.generativeai as genai

# ============================================================
# GEMINI AI CONFIGURATION (Used for Chatbot only)
# ============================================================
# Set GEMINI_API_KEY in your environment. The old key is kept only as a fallback so the chatbot keeps working;
# please rotate it, since it was committed in source code.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyA7N2IsdchHxteEgkW81hf0bpgtoUWqPvs')
GEMINI_MODEL_NAME = 'gemini-2.0-flash'

# Initialize Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print(f"[Gemini] Initialized model: {GEMINI_MODEL_NAME}")
except Exception as e:
    gemini_model = None
    print(f"[Gemini] Failed to initialize: {e}")
# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///legal_documents.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Set timezone to IST (Indian Standard Time)
IST = timezone(timedelta(hours=5, minutes=30))

def get_current_time():
    """Get current time in IST"""
    return datetime.now(IST)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/audio', exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

translator = Translator()

# Initialize ML-based simplifier
simplifier = LegalTextSimplifier()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

ocr_reader = None

# ============================================================
# GEMINI HELPER FOR CHATBOT
# ============================================================
def ask_gemini(question, context=None):
    """
    Ask Gemini a legal question and return a clean response.
    Returns (response_text, error_message).
    """
    if gemini_model is None:
        return None, "Gemini AI is not configured. Please check the API key."

    try:
        # System-style context to constrain Gemini's behavior
        system_context = (
            "You are 'LegalEase Assistant', an AI legal information chatbot for an "
            "Indian legal-tech platform. Your job is to provide clear, accurate, "
            "easy-to-understand legal information to common users.\n\n"
            "Rules you MUST follow:\n"
            "1. Give general legal information, NOT personalized legal advice.\n"
            "2. Write in simple, plain English that a non-lawyer can understand.\n"
            "3. Use short paragraphs. Do NOT use markdown symbols like **, *, #, -, or bullet characters.\n"
            "4. Do NOT use code blocks, tables, or special formatting.\n"
            "5. Where relevant, mention Indian laws (IPC, CrPC, Contract Act, etc.).\n"
            "6. If the user's situation is serious or specific, recommend consulting a qualified lawyer.\n"
            "7. Always end with a short disclaimer that this is general information, not legal advice.\n"
            "8. Keep the answer under 250 words unless the user asks for detail.\n"
        )

        if context:
            system_context += f"\nExtra context from previous exchange:\n{context}\n"

        full_prompt = (
            f"{system_context}\n"
            f"User's legal question:\n{question}\n\n"
            f"Your plain-English answer:"
        )

        chat = gemini_model.start_chat(history=[])
        gemini_response = chat.send_message(full_prompt)

        raw_text = ""
        if hasattr(gemini_response, "text") and gemini_response.text:
            raw_text = gemini_response.text
        else:
            # Fallback: reconstruct from candidates
            try:
                raw_text = gemini_response.candidates[0].content.parts[0].text
            except Exception:
                raw_text = ""

        if not raw_text:
            return None, "Gemini returned an empty response."

        # Clean markdown / special characters for plain display
        cleaned = clean_text_for_display(raw_text)
        return cleaned, None

    except Exception as e:
        print(f"[Gemini] Error while asking: {e}")
        return None, f"Gemini request failed: {str(e)}"
def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(['en'], gpu=False)
    return ocr_reader

LANGUAGE_MAP = {
    'hi': 'hi', 'te': 'te', 'ta': 'ta', 'kn': 'kn', 'ml': 'ml',
    'bn': 'bn', 'mr': 'mr', 'gu': 'gu', 'pa': 'pa', 'en': 'en',
    'es': 'es', 'fr': 'fr', 'de': 'de', 'zh-cn': 'zh-cn', 'ja': 'ja',
    'ru': 'ru', 'ar': 'ar', 'pt': 'pt', 'it': 'it'
}

LANGUAGE_NAMES = {
    'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil', 'kn': 'Kannada',
    'ml': 'Malayalam', 'bn': 'Bengali', 'mr': 'Marathi', 'gu': 'Gujarati',
    'pa': 'Punjabi', 'en': 'English', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'zh-cn': 'Chinese', 'ja': 'Japanese', 'ru': 'Russian',
    'ar': 'Arabic', 'pt': 'Portuguese', 'it': 'Italian'
}

DOCUMENT_TYPE_PATTERNS = {
    'Contract Agreement': [r'contract', r'agreement', r'terms and conditions', r'service agreement'],
    'Employment Document': [r'employment', r'employee', r'employer', r'hire', r'termination'],
    'Property/Land Document': [r'property', r'land', r'real estate', r'lease', r'rental'],
    'Financial Document': [r'loan', r'credit', r'finance', r'bank', r'investment'],
    'Family Document': [r'divorce', r'marriage', r'custody', r'child support'],
    'Corporate Document': [r'company', r'corporation', r'board', r'director'],
    'Government/Legal Notice': [r'notice', r'order', r'judgment', r'decree'],
    'Intellectual Property': [r'patent', r'trademark', r'copyright'],
    'Criminal Document': [r'crime', r'criminal', r'offence', r'complaint'],
    'Civil Document': [r'civil', r'tort', r'negligence'],
    'Insurance Document': [r'insurance', r'claim', r'coverage'],
    'Education Document': [r'school', r'college', r'university'],
    'Healthcare Document': [r'medical', r'health', r'hospital']
}

LAWYER_SPECIALIZATIONS = {
    'Contract Agreement': ['Contract Law', 'Corporate Law', 'Business Law'],
    'Employment Document': ['Employment Law', 'Labor Law', 'HR Law'],
    'Property/Land Document': ['Property Law', 'Real Estate Law', 'Land Law'],
    'Financial Document': ['Banking Law', 'Finance Law', 'Tax Law'],
    'Family Document': ['Family Law', 'Divorce Law', 'Succession Law'],
    'Corporate Document': ['Corporate Law', 'Company Law', 'Securities Law'],
    'Government/Legal Notice': ['Constitutional Law', 'Administrative Law'],
    'Intellectual Property': ['Intellectual Property Law', 'Patent Law'],
    'Criminal Document': ['Criminal Law', 'Criminal Procedure'],
    'Civil Document': ['Civil Law', 'Tort Law', 'Litigation'],
    'Insurance Document': ['Insurance Law', 'Contract Law'],
    'Education Document': ['Education Law', 'Administrative Law'],
    'Healthcare Document': ['Healthcare Law', 'Medical Law']
}

def clean_text_for_display(text):
    """Remove markdown and special characters for display"""
    if not text:
        return text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text_for_tts(text):
    """Clean text for TTS"""
    if not text:
        return text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\(\)]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def classify_document_type(text):
    """Classify document type (weighted detector). Returns the type name only."""
    return detect_document_type(text)['type']

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=get_current_time)
    documents = db.relationship('Document', backref='user', lazy=True)
    legal_queries = db.relationship('LegalQuery', backref='user', lazy=True)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    simplified_text = db.Column(db.Text)
    risk_score = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    document_type = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=get_current_time)
    language = db.Column(db.String(10), default='en')
    filename = db.Column(db.String(200))
    file_type = db.Column(db.String(50))
    clauses = db.relationship('Clause', backref='document', lazy=True)
    sections = db.relationship('LegalSection', backref='document', lazy=True)

class Clause(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    original_clause = db.Column(db.Text, nullable=False)
    simplified_clause = db.Column(db.Text)
    risk_score = db.Column(db.Float)
    is_important = db.Column(db.Boolean, default=False)
    clause_type = db.Column(db.String(50))

class LegalExpert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    specialization = db.Column(db.String(200), nullable=False)
    experience = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Float, default=4.5)
    location = db.Column(db.String(200))
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    address = db.Column(db.String(500))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(100))
    fee_range = db.Column(db.String(50))
    is_available = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=get_current_time)

class LegalQuery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=get_current_time)
    is_resolved = db.Column(db.Boolean, default=False)

class LegalSection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    section_type = db.Column(db.String(20))
    section_number = db.Column(db.String(50))
    description = db.Column(db.Text)
    context = db.Column(db.Text)

class Deadline(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=True)
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50))
    priority = db.Column(db.String(10), default='medium')
    due_date = db.Column(db.Date, nullable=True)
    period_text = db.Column(db.String(50))
    context = db.Column(db.Text)
    remind_on = db.Column(db.Date, nullable=True)
    remind_days_before = db.Column(db.Integer, nullable=True)
    is_done = db.Column(db.Boolean, default=False)
    is_custom = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=get_current_time)

    def to_dict(self):
        today = get_current_time().date()
        d = (self.due_date - today).days if self.due_date else None
        return {
            'id': self.id, 'document_id': self.document_id, 'title': self.title,
            'category': self.category, 'priority': self.priority,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'period_text': self.period_text, 'context': self.context,
            'remind_on': self.remind_on.isoformat() if self.remind_on else None,
            'remind_days_before': self.remind_days_before,
            'is_done': self.is_done, 'days_left': d,
            'status': deadline_status(self, today),
        }


def deadline_status(dl, today):
    if dl.is_done:
        return 'done'
    if dl.due_date is None:
        return 'undated'
    d = (dl.due_date - today).days
    if d < 0 and dl.category in ('Start / Effective Date', 'Important Date'):
        return 'past'   # e.g. the date the agreement was signed - not an overdue obligation
    if d < 0:
        return 'overdue'
    if d <= 7:
        return 'urgent'
    if d <= 30:
        return 'soon'
    return 'upcoming'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# File handling functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    try:
        text = ""
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return extract_text_from_image_pdf(filepath)

def extract_text_from_image_pdf(filepath):
    try:
        images = pdf2image.convert_from_path(filepath, dpi=300)
        text = ""
        reader = get_ocr_reader()
        for image in images:
            img_array = np.array(image)
            results = reader.readtext(img_array)
            for result in results:
                text += result[1] + " "
            text += "\n"
        return text.strip()
    except Exception as e:
        print(f"PDF OCR extraction error: {e}")
        return ""

def extract_text_from_docx(filepath):
    try:
        doc = docx.Document(filepath)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_txt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as file:
                return file.read().strip()
        except Exception as e:
            print(f"TXT extraction error: {e}")
            return ""

def extract_text_from_image(filepath):
    try:
        reader = get_ocr_reader()
        results = reader.readtext(filepath)
        text = ""
        for result in results:
            text += result[1] + " "
        return text.strip()
    except Exception as e:
        print(f"Image OCR error: {e}")
        try:
            image = Image.open(filepath)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e2:
            print(f"Tesseract OCR error: {e2}")
            return ""

def extract_text_from_file(filepath, filename):
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if file_ext == 'pdf':
        return extract_text_from_pdf(filepath)
    elif file_ext in ['docx', 'doc']:
        return extract_text_from_docx(filepath)
    elif file_ext == 'txt':
        return extract_text_from_txt(filepath)
    elif file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']:
        return extract_text_from_image(filepath)
    return ""

def extract_clauses(text):
    """Extract legal clauses from text"""
    clauses = []
    clause_patterns = [
        r'(?i)(this\s+agreement|section|clause|article)\s+(\d+\.?\d*)\s*[:.]',
        r'(?i)(party|parties|company|client)\s+(shall|will|must|may)',
        r'(?i)(indemnify|liability|warranty|represent|guarantee)',
        r'(?i)(payment|fee|charge|cost|expense)',
        r'(?i)(termination|cancellation|renewal)',
        r'(?i)(confidentiality|non-disclosure|privacy)',
        r'(?i)(dispute|arbitration|governing\s+law|jurisdiction)',
        r'(?i)(intellectual\s+property|copyright|trademark)'
    ]
    
    sentences = sent_tokenize(text)
    current_clause = []
    clause_number = 1
    
    for sentence in sentences:
        is_start = any(re.search(pattern, sentence) for pattern in clause_patterns)
        if is_start and current_clause:
            clauses.append({
                'number': clause_number,
                'text': ' '.join(current_clause),
                'type': classify_clause_type(' '.join(current_clause))
            })
            clause_number += 1
            current_clause = [sentence]
        else:
            current_clause.append(sentence)
    
    if current_clause:
        clauses.append({
            'number': clause_number,
            'text': ' '.join(current_clause),
            'type': classify_clause_type(' '.join(current_clause))
        })
    return clauses

def classify_clause_type(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ['indemnify', 'liability']):
        return 'liability'
    elif any(word in text_lower for word in ['payment', 'fee', 'charge']):
        return 'payment'
    elif any(word in text_lower for word in ['termination', 'cancel', 'renew']):
        return 'termination'
    elif any(word in text_lower for word in ['confidential', 'non-disclosure', 'privacy']):
        return 'confidentiality'
    elif any(word in text_lower for word in ['dispute', 'arbitration', 'law']):
        return 'dispute'
    elif any(word in text_lower for word in ['intellectual', 'copyright', 'trademark', 'patent']):
        return 'intellectual_property'
    return 'general'

def calculate_risk_score(text, clauses):
    risk_score = 0.0
    risk_keywords = {
        'unlimited liability': 0.15, 'indemnify': 0.15, 'indemnification': 0.15,
        'penalty': 0.15, 'forfeiture': 0.15, 'liquidated damages': 0.15,
        'termination without cause': 0.15, 'confidentiality breach': 0.15,
        'non-compete': 0.15, 'exclusive': 0.12, 'irrevocable': 0.12,
        'waiver': 0.12, 'liability': 0.10, 'damages': 0.10, 'breach': 0.10,
        'default': 0.10, 'arbitration': 0.10, 'governing law': 0.10,
        'jurisdiction': 0.10, 'termination': 0.10, 'cancellation': 0.10,
        'renewal': 0.08, 'confidential': 0.08, 'dispute': 0.08,
        'warranty': 0.08, 'representation': 0.08, 'payment': 0.05,
        'fee': 0.05, 'charge': 0.05, 'cost': 0.05, 'expense': 0.05,
        'notice': 0.05, 'period': 0.05, 'agreement': 0.03,
        'party': 0.03, 'parties': 0.03,
    }
    
    text_lower = text.lower()
    for keyword, weight in risk_keywords.items():
        if keyword in text_lower:
            risk_score += weight
    
    for clause in clauses:
        if clause['type'] == 'liability':
            risk_score += 0.15
        elif clause['type'] == 'payment':
            risk_score += 0.10
        elif clause['type'] == 'termination':
            risk_score += 0.10
        elif clause['type'] == 'confidentiality':
            risk_score += 0.08
        elif clause['type'] == 'dispute':
            risk_score += 0.08
        elif clause['type'] == 'intellectual_property':
            risk_score += 0.08
    
    return min(risk_score, 1.0)

def get_risk_level(score):
    if score < 0.3:
        return 'Low'
    elif score < 0.6:
        return 'Medium'
    return 'High'

def translate_text(text, target_lang='hi'):
    try:
        if target_lang == 'en':
            return text
        if len(text) > 5000:
            chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
            translated_chunks = []
            for chunk in chunks:
                translated = translator.translate(chunk, dest=target_lang)
                translated_chunks.append(translated.text)
            return ' '.join(translated_chunks)
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def generate_multilingual_tts(text, lang_code='en'):
    try:
        clean_text = clean_text_for_tts(text)
        tts_text = clean_text[:1000] if len(clean_text) > 1000 else clean_text
        if not tts_text.strip():
            return None
        if lang_code != 'en' and lang_code in LANGUAGE_MAP:
            translated_text = translate_text(tts_text, lang_code)
            tts_text = translated_text
        gtts_lang = LANGUAGE_MAP.get(lang_code, 'en')
        tts = gTTS(text=tts_text, lang=gtts_lang, slow=False)
        filename = f"tts_{lang_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        filepath = os.path.join('static', 'audio', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tts.save(filepath)
        return filename
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def generate_tts(text, lang='en'):
    return generate_multilingual_tts(text, lang)

def extract_legal_sections(text):
    sections = []
    patterns = {
        'IPC': [r'IPC\s*(?:section|sec\.?)?\s*(\d+[A-Z]?)', r'section\s*(\d+[A-Z]?)\s*of\s*(?:the\s*)?IPC'],
        'CRPC': [r'CRPC\s*(?:section|sec\.?)?\s*(\d+[A-Z]?)', r'section\s*(\d+[A-Z]?)\s*of\s*(?:the\s*)?CRPC'],
        'SECTION': [r'section\s*(\d+[A-Z]?)[\s\-\(\)]', r'Sec\.?\s*(\d+[A-Z]?)'],
        'ARTICLE': [r'Article\s*(\d+[A-Z]?)', r'Art\.\s*(\d+[A-Z]?)']
    }
    
    for pattern_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if groups:
                    sections.append({
                        'type': pattern_type,
                        'number': str(groups[0]),
                        'context': text[max(0, match.start()-60):min(len(text), match.end()+60)]
                    })
    
    unique_sections = []
    seen = set()
    for section in sections:
        key = f"{section['type']}_{section['number']}"
        if key not in seen:
            seen.add(key)
            unique_sections.append(section)
    return unique_sections

def get_section_description(section_type, section_number):
    descriptions = {
        'IPC': {
            '302': 'Punishment for murder - death or imprisonment for life',
            '420': 'Cheating and dishonestly inducing delivery of property',
            '498A': 'Cruelty by husband or relatives of husband',
            '376': 'Punishment for rape',
        },
        'SECTION': {
            '1': 'General provision - scope and application',
            '2': 'Definitions and interpretations',
        }
    }
    desc = descriptions.get(section_type, {}).get(str(section_number))
    return desc if desc else 'Legal provision - consult a lawyer for exact interpretation'

def calculate_distance(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return float('inf')
    R = 6371
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def geocode_location(location_name):
    try:
        location_name = location_name.strip()
        if not location_name:
            return None, None
        if 'bangalore' not in location_name.lower() and 'bengaluru' not in location_name.lower():
            location_name = f"{location_name}, Bangalore, Karnataka, India"
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': location_name, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': 'LegalDocSimplifier/1.0'}
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        if data and len(data) > 0:
            return float(data[0]['lat']), float(data[0]['lon'])
        return None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None

def initialize_experts():
    """
    Sample/random lawyers are no longer seeded. Lawyers now come from live
    location-based search (OpenStreetMap). The LegalExpert table is kept so you can add
    real, verified lawyers (with latitude/longitude) which are ranked by real distance.
    Previously seeded demo rows are removed.
    """
    try:
        LegalExpert.query.filter(LegalExpert.email.like('%@legalexpert.in')).delete(synchronize_session=False)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"[init] could not clean demo experts: {e}")


def build_analysis(text, document_type=None, risk_level='Medium'):
    """Compute detection details + personalised risk analysis for a document text."""
    detection = detect_document_type(text)
    dtype = document_type or detection['type']
    risk = analyze_risks(text, dtype, risk_level)
    profile = get_specialization_profile(dtype)
    return detection, risk, profile


def save_deadlines(document, text):
    """Extract deadlines from text and store them for the user (replaces previous auto ones)."""
    Deadline.query.filter_by(document_id=document.id, is_custom=False).delete()
    for d in extract_deadlines(text, get_current_time().date()):
        due = datetime.strptime(d['date'], '%Y-%m-%d').date() if d['date'] else None
        db.session.add(Deadline(
            user_id=document.user_id, document_id=document.id, title=d['title'],
            category=d['category'], priority=d['priority'], due_date=due,
            period_text=d['period_text'], context=d['context'][:1000]))
    db.session.commit()


def migrate_db():
    """Create new tables (Deadline). Existing tables are untouched."""
    db.create_all()


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    documents = Document.query.filter_by(user_id=current_user.id).order_by(Document.created_at.desc()).all()
    total_docs = len(documents)
    high_risk_docs = sum(1 for doc in documents if doc.risk_level == 'High')
    avg_risk = sum(doc.risk_score or 0 for doc in documents) / total_docs if total_docs > 0 else 0
    doc_types = {}
    for doc in documents:
        if doc.document_type:
            doc_types[doc.document_type] = doc_types.get(doc.document_type, 0) + 1
    upcoming = [d.to_dict() for d in Deadline.query.filter_by(user_id=current_user.id, is_done=False)
                .filter(Deadline.due_date.isnot(None)).order_by(Deadline.due_date).all()]
    upcoming = [d for d in upcoming if d['status'] != 'past'][:6]
    return render_template('dashboard.html', documents=documents, total_docs=total_docs,
                         high_risk_docs=high_risk_docs, avg_risk=avg_risk, doc_types=doc_types,
                         upcoming_deadlines=upcoming)

@app.route('/simplify', methods=['GET', 'POST'])
@login_required
def simplify():
    if request.method == 'POST':
        extracted_text = ""
        filename = None
        file_type = None
        
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                extracted_text = extract_text_from_file(filepath, filename)
                file_type = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
                try:
                    os.remove(filepath)
                except:
                    pass
                if not extracted_text:
                    flash('Could not extract text from the file.', 'warning')
                    return redirect(url_for('simplify'))
        
        text_input = request.form.get('legal_text', '').strip()
        if text_input:
            extracted_text = text_input
        
        if not extracted_text:
            flash('Please provide legal text or upload a file', 'warning')
            return redirect(url_for('simplify'))
        
        document_type = classify_document_type(extracted_text)
        clauses = extract_clauses(extracted_text)
        
        # Use ML-based simplification
        simplified_clauses = []
        for clause in clauses:
            simplified_text = simplifier.simplify(clause['text'])
            simplified_clauses.append({
                'original': clause['text'],
                'simplified': simplified_text,
                'type': clause['type']
            })
        
        risk_score = calculate_risk_score(extracted_text, clauses)
        risk_level = get_risk_level(risk_score)
        
        # Use ML-based simplification for full text
        simplified_full = simplifier.simplify(extracted_text)
        
        new_document = Document(
            user_id=current_user.id,
            original_text=extracted_text,
            simplified_text=simplified_full,
            risk_score=risk_score,
            risk_level=risk_level,
            document_type=document_type,
            filename=filename,
            file_type=file_type
        )
        db.session.add(new_document)
        db.session.commit()
        
        for clause_data in simplified_clauses:
            new_clause = Clause(
                document_id=new_document.id,
                original_clause=clause_data['original'],
                simplified_clause=clause_data['simplified'],
                clause_type=clause_data['type'],
                is_important=clause_data['type'] in ['liability', 'payment', 'termination']
            )
            db.session.add(new_clause)
        db.session.commit()
        
        legal_sections = extract_legal_sections(extracted_text)
        for section in legal_sections:
            section_desc = get_section_description(section['type'], section['number'])
            new_section = LegalSection(
                document_id=new_document.id,
                section_type=section['type'],
                section_number=section['number'],
                description=section_desc,
                context=section['context'][:500]
            )
            db.session.add(new_section)
        db.session.commit()
        
        tts_file = generate_tts(simplified_full, 'en')
        detection, risk_analysis, spec_profile = build_analysis(extracted_text, document_type, risk_level)
        save_deadlines(new_document, extracted_text)
        deadlines = [d.to_dict() for d in Deadline.query.filter_by(document_id=new_document.id)
                     .order_by(Deadline.due_date.is_(None), Deadline.due_date).all()]
        lawyer_specializations = [spec_profile['primary']] + spec_profile['related']
        
        return render_template('result.html',
                             detection=detection, risk_analysis=risk_analysis,
                             spec_profile=spec_profile, deadlines=deadlines,
                             original_text=extracted_text,
                             simplified_text=simplified_full,
                             clauses=simplified_clauses,
                             risk_score=risk_score,
                             risk_level=risk_level,
                             document_type=document_type,
                             tts_file=tts_file,
                             doc_id=new_document.id,
                             filename=filename,
                             file_type=file_type,
                             legal_sections=legal_sections,
                             lawyer_specializations=lawyer_specializations,
                             language_names=LANGUAGE_NAMES)
    
    return render_template('simplify.html')

@app.route('/history')
@login_required
def history():
    documents = Document.query.filter_by(user_id=current_user.id).order_by(Document.created_at.desc()).all()
    return render_template('history.html', documents=documents)

@app.route('/document/<int:doc_id>')
@login_required
def view_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    if document.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    clauses = Clause.query.filter_by(document_id=doc_id).all()
    sections = LegalSection.query.filter_by(document_id=doc_id).all()
    
    legal_sections = [{'type': s.section_type, 'number': s.section_number,
                       'description': s.description, 'context': s.context} for s in sections]
    detection, risk_analysis, spec_profile = build_analysis(
        document.original_text, document.document_type, document.risk_level or 'Medium')
    lawyer_specializations = [spec_profile['primary']] + spec_profile['related']
    clause_list = [{'original': c.original_clause, 'simplified': c.simplified_clause,
                   'type': c.clause_type} for c in clauses]
    if Deadline.query.filter_by(document_id=doc_id).count() == 0:
        save_deadlines(document, document.original_text)
    deadlines = [d.to_dict() for d in Deadline.query.filter_by(document_id=doc_id)
                 .order_by(Deadline.due_date.is_(None), Deadline.due_date).all()]
    
    return render_template('result.html',
                         detection=detection, risk_analysis=risk_analysis,
                         spec_profile=spec_profile, deadlines=deadlines,
                         original_text=document.original_text,
                         simplified_text=document.simplified_text,
                         clauses=clause_list,
                         risk_score=document.risk_score,
                         risk_level=document.risk_level,
                         document_type=document.document_type,
                         doc_id=doc_id,
                         filename=document.filename,
                         file_type=document.file_type,
                         legal_sections=legal_sections,
                         lawyer_specializations=lawyer_specializations,
                         language_names=LANGUAGE_NAMES)

@app.route('/find_lawyers/<int:doc_id>', methods=['POST'])
@login_required
def find_lawyers(doc_id):
    """
    Location-based lawyer recommendation.
    Body (JSON): {latitude, longitude, radius_km?}
    The browser asks the user for permission and sends the coordinates.
    """
    document = Document.query.get_or_404(doc_id)
    if document.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403

    payload = request.get_json(silent=True) or {}
    try:
        lat = float(payload.get('latitude'))
        lon = float(payload.get('longitude'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Valid latitude and longitude are required.'}), 400
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({'error': 'Coordinates out of range.'}), 400
    try:
        radius_km = max(1, min(float(payload.get('radius_km', 10)), 50))
    except (TypeError, ValueError):
        radius_km = 10

    profile = get_specialization_profile(document.document_type or 'General Legal Document')
    primary, related = profile['primary'], profile['related']

    candidates = []
    # 1) curated directory (distance computed from real coordinates)
    for e in LegalExpert.query.filter_by(is_available=True).all():
        if e.latitude is None or e.longitude is None:
            continue
        item = lawyer_finder.expert_to_dict(e, lat, lon, primary, related)
        if item['distance_km'] is not None and item['distance_km'] <= radius_km * 3:
            candidates.append(item)

    # 2) live web sources (Google Places if key set, OpenStreetMap Overpass, OpenStreetMap search)
    google_key = os.environ.get('GOOGLE_PLACES_API_KEY')
    live, source_status, source_errors = lawyer_finder.collect_lawyers(lat, lon, radius_km, primary, related, google_key)
    if not live and any(v == 'ok' for v in source_status.values()) and radius_km < 25:
        radius_km = min(radius_km * 2.5, 50)   # nothing found: widen once
        live, source_status, source_errors = lawyer_finder.collect_lawyers(lat, lon, radius_km, primary, related, google_key)
    candidates.extend(live)
    osm_status = 'ok' if any(v == 'ok' for v in source_status.values()) else 'unavailable'
    if source_errors:
        print('[find_lawyers] source errors:', source_errors)

    ranked = lawyer_finder.rank_lawyers(candidates)[:12]
    return jsonify({
        'document_type': document.document_type,
        'required_specialization': primary,
        'related_specializations': related,
        'user_location': {'latitude': lat, 'longitude': lon,
                          'accuracy_m': payload.get('accuracy'),
                          **{k: v for k, v in lawyer_finder.reverse_geocode_full(lat, lon).items()}},
        'radius_km': round(radius_km, 1),
        'osm_status': osm_status,
        'source_status': source_status,
        'source_errors': source_errors[:6],
        'lawyers': ranked,
    })


@app.route('/get_experts/<int:doc_id>')
@login_required
def get_experts(doc_id):
    """Backward-compatible endpoint: tells the client to use the location-based flow."""
    document = Document.query.get_or_404(doc_id)
    if document.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    profile = get_specialization_profile(document.document_type or 'General Legal Document')
    return jsonify({'experts': [], 'location_required': True,
                    'required_specialization': profile['primary']})


# ============================================================
# DEADLINES & REMINDERS
# ============================================================
def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d').date()
    except ValueError:
        return None


def _safe_next():
    nxt = request.form.get('next') or ''
    return nxt if nxt.startswith('/') and not nxt.startswith('//') else None


@app.route('/geocode_place')
@login_required
def geocode_place():
    """Fallback when the user does not share GPS: turn a typed area/city into coordinates."""
    q = (request.args.get('q') or '').strip()
    if not q:
        return jsonify({'error': 'Empty query'}), 400
    try:
        r = requests.get('https://nominatim.openstreetmap.org/search', timeout=8,
                         headers=lawyer_finder.HEADERS,
                         params={'q': q, 'format': 'json', 'limit': 1})
        data = r.json() if r.ok else []
        if data:
            return jsonify({'latitude': float(data[0]['lat']), 'longitude': float(data[0]['lon']),
                            'label': data[0].get('display_name')})
    except Exception as e:
        print(f'[geocode_place] {e}')
    return jsonify({'latitude': None, 'longitude': None})


@app.route('/deadlines')
@login_required
def deadlines_page():
    items = Deadline.query.filter_by(user_id=current_user.id).order_by(
        Deadline.is_done, Deadline.due_date.is_(None), Deadline.due_date).all()
    data = [d.to_dict() for d in items]
    doc_names = {d.id: (d.document_type or 'Document') + f' #{d.id}' for d in
                 Document.query.filter_by(user_id=current_user.id).all()}
    for d in data:
        d['document_name'] = doc_names.get(d['document_id'])
    return render_template('deadlines.html', deadlines=data,
                           today=get_current_time().date().isoformat())


@app.route('/deadline/add', methods=['POST'])
@login_required
def add_deadline():
    title = (request.form.get('title') or '').strip()
    if not title:
        flash('Please enter a title for the deadline.', 'warning')
        return redirect(_safe_next() or url_for('deadlines_page'))
    due = _parse_date(request.form.get('due_date'))
    dl = Deadline(user_id=current_user.id, title=title[:200],
                  category=request.form.get('category') or 'Custom', priority='medium',
                  due_date=due, context=(request.form.get('notes') or '')[:1000], is_custom=True)
    doc_id = request.form.get('document_id', type=int)
    if doc_id and Document.query.filter_by(id=doc_id, user_id=current_user.id).first():
        dl.document_id = doc_id
    days_before = request.form.get('remind_days_before', type=int)
    if due and days_before is not None:
        dl.remind_days_before = days_before
        dl.remind_on = due - timedelta(days=days_before)
    db.session.add(dl)
    db.session.commit()
    flash('Deadline added.', 'success')
    return redirect(_safe_next() or url_for('deadlines_page'))


@app.route('/deadline/<int:dl_id>/update', methods=['POST'])
@login_required
def update_deadline(dl_id):
    """Set/modify due date and reminder. JSON or form."""
    dl = Deadline.query.get_or_404(dl_id)
    if dl.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    data = request.get_json(silent=True) or request.form
    if 'due_date' in data:
        dl.due_date = _parse_date(data.get('due_date'))
    days_before = data.get('remind_days_before')
    custom_remind = _parse_date(data.get('remind_on'))
    if custom_remind:
        dl.remind_on = custom_remind
        dl.remind_days_before = (dl.due_date - custom_remind).days if dl.due_date else None
    elif days_before not in (None, ''):
        try:
            n = int(days_before)
        except ValueError:
            return jsonify({'error': 'Invalid reminder'}), 400
        if not dl.due_date:
            return jsonify({'error': 'Set a due date first.'}), 400
        dl.remind_days_before = n
        dl.remind_on = dl.due_date - timedelta(days=n)
    elif data.get('clear_reminder'):
        dl.remind_on = None
        dl.remind_days_before = None
    db.session.commit()
    return jsonify({'success': True, 'deadline': dl.to_dict()})


@app.route('/deadline/<int:dl_id>/toggle', methods=['POST'])
@login_required
def toggle_deadline(dl_id):
    dl = Deadline.query.get_or_404(dl_id)
    if dl.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    dl.is_done = not dl.is_done
    db.session.commit()
    return jsonify({'success': True, 'is_done': dl.is_done})


@app.route('/deadline/<int:dl_id>/delete', methods=['POST'])
@login_required
def delete_deadline(dl_id):
    dl = Deadline.query.get_or_404(dl_id)
    if dl.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    db.session.delete(dl)
    db.session.commit()
    return jsonify({'success': True})


@app.route('/deadline/<int:dl_id>/calendar.ics')
@login_required
def deadline_ics(dl_id):
    """Download an .ics event so the deadline can go into Google/Apple/Outlook calendar."""
    dl = Deadline.query.get_or_404(dl_id)
    if dl.user_id != current_user.id or not dl.due_date:
        flash('This deadline has no date yet.', 'warning')
        return redirect(url_for('deadlines_page'))
    start = dl.due_date.strftime('%Y%m%d')
    end = (dl.due_date + timedelta(days=1)).strftime('%Y%m%d')
    summary = (dl.title or 'Deadline').replace('\n', ' ')
    desc = (dl.context or '').replace('\n', ' ').replace(',', '\\,')
    alarm = ''
    if dl.remind_days_before:
        alarm = (f"BEGIN:VALARM\r\nTRIGGER:-P{int(dl.remind_days_before)}D\r\nACTION:DISPLAY\r\n"
                 f"DESCRIPTION:{summary}\r\nEND:VALARM\r\n")
    ics = ("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//LegalEase//Deadlines//EN\r\nBEGIN:VEVENT\r\n"
           f"UID:legalease-deadline-{dl.id}@legalease\r\n"
           f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}\r\n"
           f"DTSTART;VALUE=DATE:{start}\r\nDTEND;VALUE=DATE:{end}\r\n"
           f"SUMMARY:LegalEase: {summary}\r\nDESCRIPTION:{desc}\r\n{alarm}END:VEVENT\r\nEND:VCALENDAR\r\n")
    resp = make_response(ics)
    resp.headers['Content-Type'] = 'text/calendar; charset=utf-8'
    resp.headers['Content-Disposition'] = f'attachment; filename=deadline_{dl.id}.ics'
    return resp


@app.route('/api/upcoming_deadlines')
@login_required
def api_upcoming_deadlines():
    """Used by the navbar badge and browser notifications."""
    today = get_current_time().date()
    items = Deadline.query.filter_by(user_id=current_user.id, is_done=False).filter(
        Deadline.due_date.isnot(None)).order_by(Deadline.due_date).all()
    upcoming, due_reminders = [], []
    for d in items:
        info = d.to_dict()
        if info['status'] == 'past':
            continue
        if info['days_left'] is not None and info['days_left'] <= 30:
            upcoming.append(info)
        if d.remind_on and d.remind_on <= today and d.due_date >= today - timedelta(days=30):
            due_reminders.append(info)
    return jsonify({'upcoming': upcoming[:10], 'due_reminders': due_reminders,
                    'count': len([u for u in upcoming if u['days_left'] <= 7])})


@app.context_processor
def inject_deadline_badge():
    """Count of urgent deadlines (<=7 days or overdue) for the navbar."""
    try:
        if current_user.is_authenticated:
            today = get_current_time().date()
            n = Deadline.query.filter_by(user_id=current_user.id, is_done=False).filter(
                Deadline.due_date.isnot(None),
                Deadline.due_date <= today + timedelta(days=7)).filter(
                ~Deadline.category.in_(['Start / Effective Date', 'Important Date']) |
                (Deadline.due_date >= today)).count()
            return {'urgent_deadline_count': n}
    except Exception:
        pass
    return {'urgent_deadline_count': 0}


@app.route('/generate_tts/<int:doc_id>/<lang>')
@login_required
def generate_tts_route(doc_id, lang):
    document = Document.query.get_or_404(doc_id)
    if document.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    tts_file = generate_multilingual_tts(document.simplified_text, lang)
    if tts_file:
        return jsonify({'tts_file': tts_file})
    return jsonify({'error': 'Failed to generate TTS'}), 500

@app.route('/translate/<int:doc_id>/<lang>')
@login_required
def translate_document(doc_id, lang):
    document = Document.query.get_or_404(doc_id)
    if document.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    try:
        translated_text = translate_text(document.simplified_text, lang)
        return jsonify({'translated_text': clean_text_for_display(translated_text)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_document/<int:doc_id>', methods=['POST'])
@login_required
def delete_document(doc_id):
    document = Document.query.get_or_404(doc_id)
    if document.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    Clause.query.filter_by(document_id=doc_id).delete()
    LegalSection.query.filter_by(document_id=doc_id).delete()
    Deadline.query.filter_by(document_id=doc_id).delete()
    db.session.delete(document)
    db.session.commit()
    return jsonify({'success': True})
# ============================================================
# DOWNLOAD ROUTES
# ============================================================

@app.route('/download/<int:doc_id>/<file_format>')
@login_required
def download_document(doc_id, file_format):
    """
    Download the simplified document in the requested format.
    Supported formats: txt, pdf, docx, html
    """
    document = Document.query.get_or_404(doc_id)

    # Security: only the owner can download
    if document.user_id != current_user.id:
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))

    # Fetch related data
    clauses = Clause.query.filter_by(document_id=doc_id).all()
    sections = LegalSection.query.filter_by(document_id=doc_id).all()

    # Prepare content pieces
    original_text = document.original_text or ''
    simplified_text = document.simplified_text or ''
    doc_type = document.document_type or 'General Legal Document'
    risk_level = document.risk_level or 'Unknown'
    risk_score = document.risk_score or 0.0

    # Timestamps in IST
    created_at = document.created_at
    if created_at:
        try:
            created_at_str = created_at.strftime('%B %d, %Y at %I:%M %p')
        except Exception:
            created_at_str = str(created_at)
    else:
        created_at_str = 'N/A'

    # Safe filename base
    safe_base = re.sub(r'[^A-Za-z0-9_\-]', '_', (doc_type or 'Legal_Document'))[:50]
    timestamp_str = datetime.now(IST).strftime('%Y%m%d_%H%M%S')
    filename_base = f"LegalEase_{safe_base}_{timestamp_str}"

    file_format = (file_format or 'txt').lower()

    # ---------------------------------------------------------
    # TXT DOWNLOAD
    # ---------------------------------------------------------
    if file_format == 'txt':
        lines = []
        lines.append("=" * 70)
        lines.append("LegalEase - Simplified Legal Document")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Document Type : {doc_type}")
        lines.append(f"Risk Level    : {risk_level}")
        lines.append(f"Risk Score    : {round(risk_score * 100)}%")
        lines.append(f"Generated On  : {created_at_str}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("SIMPLIFIED DOCUMENT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(simplified_text)
        lines.append("")
        lines.append("=" * 70)
        lines.append("ORIGINAL DOCUMENT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(original_text)
        lines.append("")

        if clauses:
            lines.append("=" * 70)
            lines.append(f"CLAUSES ({len(clauses)})")
            lines.append("=" * 70)
            for idx, c in enumerate(clauses, 1):
                lines.append("")
                lines.append(f"--- Clause {idx} [{c.clause_type or 'general'}] ---")
                lines.append("")
                lines.append("ORIGINAL:")
                lines.append(c.original_clause or '')
                lines.append("")
                lines.append("SIMPLIFIED:")
                lines.append(c.simplified_clause or '')
                lines.append("")
                lines.append("-" * 70)

        if sections:
            lines.append("")
            lines.append("=" * 70)
            lines.append(f"LEGAL SECTIONS ({len(sections)})")
            lines.append("=" * 70)
            for s in sections:
                lines.append("")
                lines.append(f"[{s.section_type}] Section {s.section_number}")
                lines.append(f"  {s.description}")
                if s.context:
                    lines.append(f"  Context: {s.context}")
                lines.append("")

        lines.append("")
        lines.append("=" * 70)
        lines.append("DISCLAIMER")
        lines.append("=" * 70)
        lines.append(
            "This document was simplified by LegalEase using machine learning. "
            "It is provided for educational purposes only and does NOT constitute "
            "legal advice. Please consult a qualified lawyer for legal matters."
        )
        lines.append("")
        lines.append("=" * 70)

        content = "\n".join(lines)
        buffer = BytesIO(content.encode('utf-8'))

        return send_file(
            buffer,
            mimetype='text/plain',
            as_attachment=True,
            download_name=f"{filename_base}.txt"
        )

    # ---------------------------------------------------------
    # PDF DOWNLOAD
    # ---------------------------------------------------------
    elif file_format == 'pdf':
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                Table, TableStyle, HRFlowable
            )
            from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY, TA_CENTER
        except ImportError:
            flash('PDF export requires reportlab. Run: pip install reportlab', 'danger')
            return redirect(url_for('view_document', doc_id=doc_id))

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
            title=f"LegalEase - {doc_type}",
            author="LegalEase"
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            fontSize=22,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=6,
            alignment=TA_CENTER
        )
        subtitle_style = ParagraphStyle(
            'SubtitleStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#6c757d'),
            alignment=TA_CENTER,
            spaceAfter=20
        )
        heading_style = ParagraphStyle(
            'HeadingStyle',
            parent=styles['Heading2'],
            fontSize=15,
            textColor=colors.HexColor('#4a90d9'),
            spaceBefore=16,
            spaceAfter=8
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['BodyText'],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY
        )
        small_style = ParagraphStyle(
            'SmallStyle',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#6c757d')
        )

        def esc(text):
            if not text:
                return ''
            return (str(text)
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;'))

        story = []

        # Cover / Header
        story.append(Paragraph("LegalEase", title_style))
        story.append(Paragraph("Simplified Legal Document", subtitle_style))

        # Info table
        risk_color_map = {
            'Low': colors.HexColor('#28a745'),
            'Medium': colors.HexColor('#ffc107'),
            'High': colors.HexColor('#dc3545')
        }
        risk_color = risk_color_map.get(risk_level, colors.grey)

        info_data = [
            ['Document Type', doc_type],
            ['Risk Level', risk_level],
            ['Risk Score', f"{round(risk_score * 100)}%"],
            ['Generated On', created_at_str],
        ]
        info_table = Table(info_data, colWidths=[4*cm, 12*cm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f7ff')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
            ('TEXTCOLOR', (1, 1), (1, 1), risk_color),
            ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))

        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor('#e0e0e0')))
        story.append(Spacer(1, 10))

        # Simplified Document
        story.append(Paragraph("Simplified Document", heading_style))
        for para in (simplified_text or '').split('\n'):
            para = para.strip()
            if para:
                story.append(Paragraph(esc(para), body_style))
                story.append(Spacer(1, 4))

        story.append(PageBreak())

        # Original Document
        story.append(Paragraph("Original Document", heading_style))
        for para in (original_text or '').split('\n'):
            para = para.strip()
            if para:
                story.append(Paragraph(esc(para), body_style))
                story.append(Spacer(1, 4))

        # Clauses
        if clauses:
            story.append(PageBreak())
            story.append(Paragraph(f"Clauses ({len(clauses)})", heading_style))
            for idx, c in enumerate(clauses, 1):
                story.append(Paragraph(
                    f"<b>Clause {idx}</b> "
                    f"<font color='#6c757d'>[{esc(c.clause_type or 'general')}]</font>",
                    body_style
                ))
                story.append(Spacer(1, 4))
                story.append(Paragraph("<b>Original:</b>", small_style))
                story.append(Paragraph(esc(c.original_clause or ''), small_style))
                story.append(Spacer(1, 4))
                story.append(Paragraph("<b>Simplified:</b>", small_style))
                story.append(Paragraph(esc(c.simplified_clause or ''), small_style))
                story.append(Spacer(1, 10))
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor('#e0e0e0')))
                story.append(Spacer(1, 6))

        # Legal Sections
        if sections:
            story.append(Spacer(1, 16))
            story.append(Paragraph(f"Legal Sections ({len(sections)})", heading_style))
            for s in sections:
                story.append(Paragraph(
                    f"<b>[{esc(s.section_type)}] Section {esc(s.section_number)}</b>",
                    body_style
                ))
                story.append(Paragraph(esc(s.description or ''), small_style))
                if s.context:
                    story.append(Paragraph(
                        f"<i>Context: {esc(s.context)}</i>", small_style
                    ))
                story.append(Spacer(1, 8))

        # Disclaimer footer
        story.append(Spacer(1, 20))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor('#e0e0e0')))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "<b>Disclaimer:</b> This document was simplified by LegalEase using machine "
            "learning and is provided for educational purposes only. It does NOT constitute "
            "legal advice. Please consult a qualified lawyer for legal matters.",
            small_style
        ))

        doc.build(story)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{filename_base}.pdf"
        )

    # ---------------------------------------------------------
    # DOCX DOWNLOAD
    # ---------------------------------------------------------
    elif file_format == 'docx':
        try:
            from docx import Document as DocxDocument
            from docx.shared import Pt, RGBColor, Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH
        except ImportError:
            flash('DOCX export requires python-docx.', 'danger')
            return redirect(url_for('view_document', doc_id=doc_id))

        buffer = BytesIO()
        docx_doc = DocxDocument()

        # Title
        h = docx_doc.add_heading('LegalEase — Simplified Legal Document', level=0)
        h.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Info paragraph
        info = docx_doc.add_paragraph()
        info.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = info.add_run(
            f"Document Type: {doc_type}  |  Risk Level: {risk_level}  |  "
            f"Risk Score: {round(risk_score * 100)}%"
        )
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x6C, 0x75, 0x7D)

        meta = docx_doc.add_paragraph()
        meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = meta.add_run(f"Generated on: {created_at_str}")
        run.font.size = Pt(9)
        run.italic = True
        run.font.color.rgb = RGBColor(0x6C, 0x75, 0x7D)

        docx_doc.add_paragraph()

        # Simplified
        docx_doc.add_heading('Simplified Document', level=1)
        for para in (simplified_text or '').split('\n'):
            para = para.strip()
            if para:
                p = docx_doc.add_paragraph(para)
                p.paragraph_format.space_after = Pt(6)

        docx_doc.add_page_break()

        # Original
        docx_doc.add_heading('Original Document', level=1)
        for para in (original_text or '').split('\n'):
            para = para.strip()
            if para:
                p = docx_doc.add_paragraph(para)
                p.paragraph_format.space_after = Pt(6)

        # Clauses
        if clauses:
            docx_doc.add_page_break()
            docx_doc.add_heading(f'Clauses ({len(clauses)})', level=1)
            for idx, c in enumerate(clauses, 1):
                docx_doc.add_heading(
                    f'Clause {idx} [{c.clause_type or "general"}]', level=2
                )
                p = docx_doc.add_paragraph()
                r = p.add_run('Original: ')
                r.bold = True
                p.add_run(c.original_clause or '')

                p = docx_doc.add_paragraph()
                r = p.add_run('Simplified: ')
                r.bold = True
                r.font.color.rgb = RGBColor(0x28, 0xA7, 0x45)
                p.add_run(c.simplified_clause or '')

        # Legal Sections
        if sections:
            docx_doc.add_heading(f'Legal Sections ({len(sections)})', level=1)
            for s in sections:
                p = docx_doc.add_paragraph()
                r = p.add_run(f'[{s.section_type}] Section {s.section_number}')
                r.bold = True
                docx_doc.add_paragraph(s.description or '')
                if s.context:
                    p = docx_doc.add_paragraph()
                    r = p.add_run(f'Context: {s.context}')
                    r.italic = True

        # Disclaimer
        docx_doc.add_paragraph()
        p = docx_doc.add_paragraph()
        r = p.add_run('Disclaimer: ')
        r.bold = True
        p.add_run(
            'This document was simplified by LegalEase using machine learning '
            'and is provided for educational purposes only. It does NOT constitute '
            'legal advice. Please consult a qualified lawyer for legal matters.'
        )

        docx_doc.save(buffer)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            as_attachment=True,
            download_name=f"{filename_base}.docx"
        )

    else:
        flash('Unsupported format. Choose TXT, PDF, or DOCX.', 'warning')
        return redirect(url_for('view_document', doc_id=doc_id))

@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    # ---------------------------------------------
    # POST: Handle user's question via Gemini AI
    # ---------------------------------------------
    if request.method == 'POST':
        question = request.form.get('query', '').strip()
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400

        # Save the question
        legal_query = LegalQuery(
            user_id=current_user.id,
            question=question
        )
        db.session.add(legal_query)
        db.session.commit()

        # Ask Gemini
        answer, error = ask_gemini(question)

        if error or not answer:
            # Fallback to ML-based answer if Gemini fails
            print(f"[Chatbot] Gemini failed ({error}); using ML fallback.")
            try:
                answer = simplifier.answer_legal_question(question)
            except Exception as e:
                print(f"[Chatbot] ML fallback also failed: {e}")
                answer = (
                    "Sorry, I couldn't process your question right now. "
                    "Please try again in a moment, or consult a qualified lawyer "
                    "for immediate assistance."
                )

        # Persist the response
        legal_query.response = answer
        legal_query.is_resolved = True
        db.session.commit()

        # Decide if we should recommend a lawyer
        need_lawyer = any(word in question.lower() for word in [
            'hire', 'need a lawyer', 'legal help', 'urgent', 'court',
            'sue', 'lawsuit', 'criminal', 'arrest', 'jail', 'prison',
            'divorce', 'custody', 'property dispute', 'police', 'fir',
            'summons', 'notice', 'eviction', 'cheque bounce', 'accident'
        ])

        # Detect legal sections mentioned or implied
        sections = extract_legal_sections(question)

        return jsonify({
            'response': answer,
            'query_id': legal_query.id,
            'need_lawyer': need_lawyer,
            'sections': sections,
            'source': 'gemini' if not error else 'ml_fallback'
        })

    # ---------------------------------------------
    # GET: Render chatbot page with history
    # ---------------------------------------------
    chat_history = LegalQuery.query.filter_by(
        user_id=current_user.id
    ).order_by(LegalQuery.created_at.desc()).limit(50).all()

    return render_template('chatbot.html', chat_history=chat_history)
@app.route('/chat_history')
@login_required
def chat_history():
    chats = LegalQuery.query.filter_by(
        user_id=current_user.id
    ).order_by(LegalQuery.created_at.desc()).limit(20).all()
    
    chat_data = []
    for chat in chats:
        sections = extract_legal_sections(chat.question)
        chat_data.append({
            'id': chat.id,
            'question': chat.question,
            'response': chat.response,
            'created_at': chat.created_at.strftime('%Y-%m-%d %H:%M'),
            'is_resolved': chat.is_resolved,
            'sections': sections
        })
    
    return jsonify({'chats': chat_data})
if __name__ == '__main__':
    with app.app_context():
        migrate_db()
        initialize_experts()
    app.run(debug=True, host='0.0.0.0', port=5000)