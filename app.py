# app.py (merged, Mac-safe, no background camera thread)
import os
import tempfile
import numpy as np
import torch
import base64
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import random
import logging
import traceback
import sqlite3
from datetime import datetime, timedelta
import json
import threading
import time

# OpenCV / Keras imports for face model
import cv2
from tensorflow.keras.models import model_from_json

from flask import Flask, request, jsonify, render_template, make_response, Response
from collections import deque

# ------------------------------
# Logging & Flask init
# ------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder=".")
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------------------
# Globals / config
# ------------------------------
# sensor file
SENSOR_JSON_PATH = "sensor_data.json"

# camera config (env override)
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FACE_MODEL_JSON = os.environ.get("FACE_MODEL_JSON", "emotiondectector.json")
FACE_MODEL_WEIGHTS = os.environ.get("FACE_MODEL_WEIGHTS", "emotiondetector.h5")

# Shared face state (kept up-to-date by capture_face_emotion or /api/facial-emotion posts)
_face_lock = threading.Lock()
latest_face_emotion = {"label": None, "confidence": 0.0, "ts": None}

# Temporal smoothing for emotion predictions (store recent predictions)
_emotion_history = deque(maxlen=5)  # Keep last 5 predictions for smoothing
_emotion_history_lock = threading.Lock()

# Video feed globals for /video_feed streaming
_video_cap = None
_video_cap_lock = threading.Lock()
_video_running = False

# cached face model (loaded lazily)
_cached_face_model = None
_cached_face_model_lock = threading.Lock()

# Wellness weights and mapping
W_AUDIO = 0.5
W_ENV = 0.3
W_PHYS = 0.2
W_FACE = 0.2

EMOTION_TO_RISK_NUM = {
    'angry': 0.90,
    'disgust': 0.80,
    'fearful': 0.85,
    'fear': 0.85,
    'sad': 0.60,
    'surprised': 0.55,
    'surprise': 0.55,
    'neutral': 0.45,
    'happy': 0.25
}

# ------------------------------
# DB init (unchanged)
# ------------------------------
def init_db():
    conn = sqlite3.connect('emotion_analyzer.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        display_name TEXT,
        email TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotion_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        emotion TEXT NOT NULL,
        confidence REAL NOT NULL,
        stress_level TEXT NOT NULL,
        recommendation TEXT,
        audio_duration REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized")

init_db()

# ------------------------------
# Audio model loading (unchanged)
# ------------------------------
print("Loading audio model...")
try:
    model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
    id2label = model.config.id2label
    print("Audio model loaded successfully")
except Exception as e:
    print(f"Error loading audio model: {e}")
    print("Using fallback audio mode")
    model = None
    feature_extractor = None
    id2label = {
        0: "angry",
        1: "disgust",
        2: "fearful",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprised"
    }

EMOTION_RECOMMENDATIONS = {
    'angry': ["Take deep breaths and count to 10","Go for a short walk to cool down","Listen to calming music","Practice progressive muscle relaxation","Write down what's making you angry"],
    'disgust': ["Focus on positive aspects of your environment","Practice mindfulness to center yourself","Move to a different location if possible","Engage in a pleasant sensory experience","Talk to someone about your feelings"],
    'fearful': ["Use grounding techniques (name 5 things you can see, 4 things you can touch, etc.)","Remember that feelings pass with time","Practice deep breathing exercises","Challenge irrational thoughts","Reach out to a supportive friend"],
    'happy': ["Savor this positive emotion fully","Share your happiness with others","Express gratitude for what's going well","Engage in activities you enjoy","Document this moment to revisit later"],
    'neutral': ["Reflect on your current goals and priorities","Use this balanced state for decision-making","Practice gratitude for your emotional balance","Set intentions for the rest of your day","Check in with your physical needs (hunger, thirst, rest)"],
    'sad': ["Allow yourself to feel without judgment","Engage in gentle physical activity","Connect with a friend or family member","Practice self-compassion","Do something creative or expressive"],
    'surprised': ["Take a moment to process the unexpected event","Focus on your breathing to center yourself","Assess whether any action is needed","Share your experience with someone","Use the energy of surprise for creativity"]
}

EMOTION_TO_STRESS = {
    'angry': 'high',
    'disgust': 'high',
    'fearful': 'high',
    'sad': 'high',
    'surprised': 'medium',
    'neutral': 'medium',
    'happy': 'low'
}

# ------------------------------
# Audio processing helpers
# ------------------------------
def load_audio(audio_path):
    try:
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        logger.info(f"Loaded audio with librosa: length={len(audio_array)}, sr={sampling_rate}")
        return audio_array, sampling_rate
    except Exception as e:
        logger.warning(f"Librosa failed to load audio: {e}")
        try:
            import soundfile as sf
            audio_array, sampling_rate = sf.read(audio_path)
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
            if sampling_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
                sampling_rate = 16000
            logger.info(f"Loaded audio with soundfile: length={len(audio_array)}, sr={sampling_rate}")
            return audio_array, sampling_rate
        except Exception as sf_e:
            logger.warning(f"SoundFile failed to load audio: {sf_e}")
            try:
                import wave
                with wave.open(audio_path, 'rb') as wf:
                    frames = wf.getnframes()
                    sample_rate = wf.getframerate()
                    sample_width = wf.getsampwidth()
                    channels = wf.getnchannels()
                    raw_data = wf.readframes(frames)
                    if sample_width == 2:
                        data = np.frombuffer(raw_data, dtype=np.int16)
                        audio_array = data.astype(np.float32) / 32768.0
                    else:
                        data = np.frombuffer(raw_data, dtype=np.int8)
                        audio_array = data.astype(np.float32) / 128.0
                    if channels > 1:
                        audio_array = audio_array.reshape(-1, channels).mean(axis=1)
                    if sample_rate != 16000:
                        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                        sample_rate = 16000
                    logger.info(f"Loaded audio with wave: length={len(audio_array)}, sr={sample_rate}")
                    return audio_array, sample_rate
            except Exception as wave_e:
                logger.error(f"All audio loading methods failed: {wave_e}")
                raise Exception("Failed to load audio file with any available method")

def preprocess_audio(audio_path, max_duration=30.0):
    try:
        audio_array, sampling_rate = load_audio(audio_path)
        max_length = int(16000 * max_duration)
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        if len(audio_array) < 8000:
            audio_array = np.pad(audio_array, (0, 8000 - len(audio_array)), 'constant')
        audio_array = librosa.util.normalize(audio_array)
        if feature_extractor:
            inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
            return inputs, audio_array
        else:
            return None, audio_array
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        traceback.print_exc()
        raise

def predict_emotion(audio_path, max_duration=30.0):
    try:
        if model is None:
            emotions = list(id2label.values())
            probs = {emotion: random.uniform(0.01, 0.2) for emotion in emotions}
            emotion = random.choice(emotions)
            probs[emotion] = random.uniform(0.6, 0.9)
            stress_level = EMOTION_TO_STRESS.get(emotion, 'medium')
            recommendation = random.choice(EMOTION_RECOMMENDATIONS.get(emotion, EMOTION_RECOMMENDATIONS['neutral']))
            return {
                'emotion': emotion,
                'confidence': probs[emotion],
                'confidence_str': f"{probs[emotion]:.2%}",
                'stress_level': stress_level,
                'recommendation': recommendation,
                'all_probabilities': probs,
                'audio_duration': 5.0,
                'note': 'Using fallback mode (model not loaded)'
            }
        inputs, audio_array = preprocess_audio(audio_path, max_duration)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_id = torch.argmax(logits, dim=-1).item()
            predicted_label = id2label[predicted_id]
            confidence = probabilities[0][predicted_id].item()
            all_probs = {id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        stress_level = EMOTION_TO_STRESS.get(predicted_label, 'medium')
        recommendations = EMOTION_RECOMMENDATIONS.get(predicted_label, EMOTION_RECOMMENDATIONS['neutral'])
        recommendation = random.choice(recommendations)
        return {
            'emotion': predicted_label,
            'confidence': confidence,
            'confidence_str': f"{confidence:.2%}",
            'stress_level': stress_level,
            'recommendation': recommendation,
            'all_probabilities': all_probs,
            'audio_duration': len(audio_array) / 16000
        }
    except Exception as e:
        logger.error(f"Error predicting emotion: {e}")
        traceback.print_exc()
        raise

# ------------------------------
# Live sensors loader (json-lines last line OR plain JSON)
# ------------------------------
def load_live_sensors():
    try:
        # support either JSON-lines last line or plain JSON file
        if not os.path.exists(SENSOR_JSON_PATH):
            return {}
        with open(SENSOR_JSON_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            # If file contains multiple lines JSON-lines style, pick last non-empty line
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            last = lines[-1]
            data = json.loads(last)
        return {
            "heart_rate": float(data.get("heartRate", data.get("heart_rate", 0))),
            "temperature": float(data.get("tempF", data.get("temperature", 0))),
            "light": int(data.get("light", 0)),
            "sound_db": float(data.get("sound", data.get("sound_db", 0)))
        }
    except Exception as e:
        logger.warning(f"load_live_sensors error: {e}")
        return {}

@app.route('/sensors', methods=['GET'])
def get_sensors():
    return jsonify(load_live_sensors())

# ------------------------------
# Face model helpers (lazy load & capture one frame)
# ------------------------------
def safe_load_face_model():
    """
    Load and cache the face model once (JSON + weights). Return None on failure.
    """
    global _cached_face_model
    with _cached_face_model_lock:
        if _cached_face_model is not None:
            return _cached_face_model
        try:
            with open(FACE_MODEL_JSON, "r", encoding="utf-8") as jf:
                model_json = jf.read()
            m = model_from_json(model_json)
            m.load_weights(FACE_MODEL_WEIGHTS)
            _cached_face_model = m
            logger.info("Face emotion model loaded (cached).")
            return _cached_face_model
        except Exception as e:
            logger.warning(f"Failed to load face model: {e}")
            _cached_face_model = None
            return None

def _enhanced_preprocess_gray(bgr_frame):
    """
    Enhanced preprocessing for better face detection and emotion recognition.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and bilateral filtering.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter for denoising while preserving edges (better than Gaussian)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    
    # CLAHE for adaptive contrast enhancement (better than simple equalizeHist)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Light Gaussian blur to reduce remaining noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

def _preprocess_roi_for_model(roi_gray):
    """
    Enhanced ROI preprocessing specifically for emotion model.
    Applies histogram equalization and proper normalization.
    """
    # Resize to model input size
    roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    
    # Apply histogram equalization for better contrast in ROI
    roi_eq = cv2.equalizeHist(roi_resized)
    
    # Normalize to [0, 1]
    roi_normalized = roi_eq.astype("float32") / 255.0
    
    # Reshape for model: (1, 48, 48, 1)
    roi_final = roi_normalized.reshape(1, 48, 48, 1)
    
    return roi_final

def _assess_frame_quality(gray_frame, face_box):
    """
    Assess frame quality for reliable emotion detection.
    Returns quality score and whether frame is acceptable.
    """
    if face_box is None:
        return 0.0, False
    
    x, y, w, h = face_box
    
    # Check face size (should be large enough)
    if w < 60 or h < 60:
        return 0.0, False
    
    # Extract face ROI
    roi = gray_frame[y:y+h, x:x+w]
    
    # Check brightness (avoid too dark or too bright)
    mean_brightness = np.mean(roi)
    if mean_brightness < 30 or mean_brightness > 220:
        return 0.3, False
    
    # Check contrast (variance)
    contrast = np.std(roi)
    if contrast < 15:
        return 0.4, False
    
    # Check blur (Laplacian variance)
    laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
    if laplacian_var < 50:  # Too blurry
        return 0.5, False
    
    # Quality score
    brightness_score = 1.0 - abs(mean_brightness - 128) / 128
    contrast_score = min(contrast / 50, 1.0)
    sharpness_score = min(laplacian_var / 200, 1.0)
    quality = (brightness_score * 0.3 + contrast_score * 0.3 + sharpness_score * 0.4)
    
    return quality, quality > 0.5

def _apply_temporal_smoothing(new_label, new_confidence):
    """
    Apply temporal smoothing using exponential moving average.
    Helps stabilize predictions and reduce flickering.
    Prevents locking onto "surprised" or other over-predicted emotions.
    """
    with _emotion_history_lock:
        # Add to history
        _emotion_history.append({
            'label': new_label,
            'confidence': new_confidence,
            'timestamp': time.time()
        })
        
        if len(_emotion_history) < 2:
            return new_label, new_confidence
        
        # Check if a single emotion dominates too much (e.g., surprised)
        label_counts = {}
        for pred in _emotion_history:
            label = pred['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # If one emotion appears in >80% of recent history, check if it's over-predicted
        total_predictions = len(_emotion_history)
        for label, count in label_counts.items():
            if count / total_predictions > 0.8 and label == 'surprise':
                # Surprised is being over-predicted, reset history to allow new predictions
                _emotion_history.clear()
                _emotion_history.append({
                    'label': new_label,
                    'confidence': new_confidence,
                    'timestamp': time.time()
                })
                return new_label, new_confidence
        
        # Count label frequencies in recent history
        label_confidences = {}
        
        for pred in _emotion_history:
            label = pred['label']
            conf = pred['confidence']
            if label not in label_confidences:
                label_confidences[label] = []
            label_confidences[label].append(conf)
        
        # Find most frequent label, but require at least 2 occurrences
        most_frequent_label = max(label_counts.items(), key=lambda x: x[1])[0]
        
        # If most frequent is "surprise" but confidence is low, prefer neutral or second most frequent
        if most_frequent_label == 'surprise':
            avg_surprise_conf = np.mean(label_confidences.get('surprise', [0]))
            # If surprise confidence is consistently low, prefer neutral
            if avg_surprise_conf < 0.5:
                # Check for neutral or other emotions
                if 'neutral' in label_counts and label_counts['neutral'] >= 1:
                    most_frequent_label = 'neutral'
                elif len(label_counts) > 1:
                    # Get second most frequent
                    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_labels) > 1:
                        most_frequent_label = sorted_labels[1][0]
        
        # Average confidence for most frequent label
        smoothed_confidence = np.mean(label_confidences.get(most_frequent_label, [new_confidence]))
        
        # Weight recent predictions more heavily (exponential decay)
        weights = np.exp(np.linspace(-1, 0, len(_emotion_history)))
        weights = weights / weights.sum()
        
        weighted_conf = 0.0
        for i, pred in enumerate(_emotion_history):
            if pred['label'] == most_frequent_label:
                weighted_conf += weights[i] * pred['confidence']
        
        smoothed_confidence = max(smoothed_confidence, weighted_conf)
        
        return most_frequent_label, smoothed_confidence

def capture_face_emotion(timeout_seconds=3, cam_index=None):
    """
    Enhanced face emotion capture with improved preprocessing and temporal smoothing.
    Returns dict: {label, confidence, ts, ...} and updates latest_face_emotion.
    """
    face_model = safe_load_face_model()
    if face_model is None:
        return {"label": None, "confidence": 0.0, "ts": None, "error": "face model not available"}

    idx = CAMERA_INDEX if cam_index is None else int(cam_index)

    # Try default backend, then CAP_DSHOW (Windows)
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        try:
            cap.release()
        except:
            pass
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

    if not cap.isOpened():
        return {"label": None, "confidence": 0.0, "ts": None, "error": "camera not available"}

    # Resolution hint
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except:
        pass

    # Warmup
    t0 = time.time()
    while time.time() - t0 < 0.5:
        cap.read()

    # Read multiple frames and pick the best quality ones
    best_candidates = []  # Store (faces, frame, gray, quality_score)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Multiple detection parameter sets for robustness
    detection_params = [
        {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (70, 70)},
        {'scaleFactor': 1.05, 'minNeighbors': 4, 'minSize': (50, 50)},
        {'scaleFactor': 1.2, 'minNeighbors': 6, 'minSize': (80, 80)},
    ]

    tries = 0
    start = time.time()
    while time.time() - start < timeout_seconds and tries < 15:
        ok, frame = cap.read()
        if not ok:
            tries += 1
            continue

        # Enhanced preprocessing
        enhanced_gray = _enhanced_preprocess_gray(frame)

        # Try multiple detection parameter sets
        for params in detection_params:
            faces = face_cascade.detectMultiScale(enhanced_gray, **params)
            if len(faces) > 0:
                # Pick largest face
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                
                # Assess quality
                quality_score, is_acceptable = _assess_frame_quality(enhanced_gray, (x, y, w, h))
                
                if is_acceptable:
                    best_candidates.append((faces, frame, enhanced_gray, quality_score))
                    break  # Found good frame with this param set

        tries += 1

    cap.release()

    if not best_candidates:
        return {"label": None, "confidence": 0.0, "ts": None, "error": "no face detected"}

    # Sort by quality score and pick top frames
    best_candidates.sort(key=lambda x: x[3], reverse=True)
    
    # Process top 3 frames and average predictions (if available)
    predictions = []
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    
    for faces, frame, gray, quality_score in best_candidates[:3]:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        
        # Extract and preprocess ROI
        roi_gray = gray[y:y+h, x:x+w]
        try:
            roi_preprocessed = _preprocess_roi_for_model(roi_gray)
        except Exception:
            continue
        
        # Predict emotion
        try:
            pred = face_model.predict(roi_preprocessed, verbose=0)[0]
            idx = int(np.argmax(pred))
            label = labels.get(idx, "neutral")
            conf = float(pred[idx])
            
            # Get all emotion confidences
            happy_conf = float(pred[3])   # index 3 is happy
            sad_conf = float(pred[5])      # index 5 is sad
            surprise_conf = float(pred[6]) # index 6 is surprise
            neutral_conf = float(pred[4])  # index 4 is neutral
            
            # Minimum confidence threshold - if max confidence is too low, default to neutral
            MIN_CONFIDENCE_THRESHOLD = 0.4
            if conf < MIN_CONFIDENCE_THRESHOLD:
                label = 'neutral'
                conf = neutral_conf if neutral_conf > 0.2 else conf
            
            # Special handling for surprised - require higher confidence to accept
            if label == 'surprise':
                # Require higher confidence for surprise (0.5) or prefer neutral/happy if close
                if conf < 0.5:
                    # If neutral or happy are close, prefer them
                    if neutral_conf > 0.3 and abs(surprise_conf - neutral_conf) < 0.15:
                        label = 'neutral'
                        conf = neutral_conf
                    elif happy_conf > 0.3 and abs(surprise_conf - happy_conf) < 0.15:
                        label = 'happy'
                        conf = happy_conf
                    else:
                        # Still low confidence, default to neutral
                        label = 'neutral'
                        conf = neutral_conf if neutral_conf > 0.2 else conf
            
            # Special handling: if happy and sad are close, prefer happy
            if abs(happy_conf - sad_conf) < 0.15 and happy_conf > 0.3:
                label = 'happy'
                conf = happy_conf
            
            # If surprise is predicted but other emotions are close, prefer the other
            if label == 'surprise' and conf < 0.6:
                # Check if neutral or happy are within 0.1
                if neutral_conf > 0.25 and abs(surprise_conf - neutral_conf) < 0.1:
                    label = 'neutral'
                    conf = neutral_conf
                elif happy_conf > 0.25 and abs(surprise_conf - happy_conf) < 0.1:
                    label = 'happy'
                    conf = happy_conf
            
            predictions.append({'label': label, 'confidence': conf, 'quality': quality_score})
        except Exception:
            continue

    if not predictions:
        return {"label": None, "confidence": 0.0, "ts": None, "error": "prediction failed"}

    # Weight predictions by quality and average
    total_weight = sum(p['quality'] for p in predictions)
    if total_weight > 0:
        weighted_label_counts = {}
        weighted_confidences = {}
        
        for p in predictions:
            weight = p['quality'] / total_weight
            label = p['label']
            conf = p['confidence']
            
            weighted_label_counts[label] = weighted_label_counts.get(label, 0) + weight
            if label not in weighted_confidences:
                weighted_confidences[label] = []
            weighted_confidences[label].append(conf * weight)
        
        # Get most weighted label
        final_label = max(weighted_label_counts.items(), key=lambda x: x[1])[0]
        final_confidence = sum(weighted_confidences.get(final_label, [0]))
        
        # Final check: if surprise is selected but confidence is low, prefer neutral
        if final_label == 'surprise' and final_confidence < 0.5:
            if 'neutral' in weighted_label_counts:
                final_label = 'neutral'
                final_confidence = sum(weighted_confidences.get('neutral', [0]))
            elif 'happy' in weighted_label_counts:
                final_label = 'happy'
                final_confidence = sum(weighted_confidences.get('happy', [0]))
    else:
        # Fallback: use most frequent label
        label_counts = {}
        for p in predictions:
            label_counts[p['label']] = label_counts.get(p['label'], 0) + 1
        final_label = max(label_counts.items(), key=lambda x: x[1])[0]
        final_confidence = np.mean([p['confidence'] for p in predictions if p['label'] == final_label])
        
        # Final check: if surprise, prefer neutral if available
        if final_label == 'surprise' and final_confidence < 0.5:
            if 'neutral' in label_counts:
                final_label = 'neutral'
                final_confidence = np.mean([p['confidence'] for p in predictions if p['label'] == 'neutral'])

    # Apply temporal smoothing
    smoothed_label, smoothed_confidence = _apply_temporal_smoothing(final_label, final_confidence)
    
    # Final safeguard: if smoothed result is surprise with low confidence, default to neutral
    if smoothed_label == 'surprise' and smoothed_confidence < 0.5:
        smoothed_label = 'neutral'
        smoothed_confidence = max(smoothed_confidence, 0.3)

    out = {"label": smoothed_label, "confidence": round(smoothed_confidence, 3), 
           "ts": datetime.utcnow().isoformat() + "Z"}

    # Update shared state
    with _face_lock:
        latest_face_emotion.update({"label": out["label"], "confidence": out["confidence"], "ts": out["ts"]})

    return out
@app.route('/capture-face', methods=['GET'])
def capture_face_route():
    cam_index = request.args.get("index")
    result = capture_face_emotion(cam_index=cam_index)
    return jsonify(result)


@app.route("/api/facial-emotion", methods=["POST"])
def facial_emotion_api():
    """
    Accept external real-time inputs (e.g., your OpenCV script can POST detected emotion).
    Example JSON: {"emotion": "happy", "confidence": 0.87}
    Storing this will also overwrite latest_face_emotion so wellness uses it.
    """
    try:
        data = request.get_json(silent=True) or {}
        emotion = data.get("emotion") or data.get("label")
        confidence = data.get("confidence", None)
        if not emotion:
            return jsonify({"error": "No emotion provided"}), 400
        ts = datetime.utcnow().isoformat() + "Z"
        with _face_lock:
            latest_face_emotion.update({
                "label": str(emotion).lower(),
                "confidence": float(confidence) if confidence is not None else latest_face_emotion.get("confidence", 0.0),
                "ts": ts
            })
        return jsonify({"status": "ok", "label": str(emotion).lower(), "confidence": confidence, "ts": ts})
    except Exception as e:
        logger.exception("facial_emotion_api error")
        return jsonify({"error": str(e)}), 500

@app.route('/face-emotion', methods=['GET'])
def face_emotion_route():
    """
    Return the most recent face emotion state (captured by /capture-face or /api/facial-emotion).
    """
    with _face_lock:
        return jsonify(latest_face_emotion)

@app.route('/video_feed')
def video_feed():
    """
    MJPEG video stream with real-time face emotion detection overlay.
    Uses enhanced preprocessing and temporal smoothing for stable predictions.
    """
    def generate_frames():
        global _video_cap, _video_running
        
        face_model = safe_load_face_model()
        if face_model is None:
            # Return error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Face model not available", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', error_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return
        
        idx = CAMERA_INDEX
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        
        detection_params = [
            {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (70, 70)},
            {'scaleFactor': 1.05, 'minNeighbors': 4, 'minSize': (50, 50)},
        ]
        
        with _video_cap_lock:
            if _video_cap is None or not _video_running:
                _video_cap = cv2.VideoCapture(idx)
                if not _video_cap.isOpened():
                    try:
                        _video_cap.release()
                    except:
                        pass
                    _video_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                
                if not _video_cap.isOpened():
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "Camera not available", (50, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    _, buffer = cv2.imencode('.jpg', error_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    return
                
                try:
                    _video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    _video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    _video_cap.set(cv2.CAP_PROP_FPS, 30)
                except:
                    pass
                
                _video_running = True
        
        frame_count = 0
        while True:
            with _video_cap_lock:
                if _video_cap is None or not _video_running:
                    break
                ret, frame = _video_cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for emotion (reduce computation)
            if frame_count % 3 == 0:
                # Enhanced preprocessing
                enhanced_gray = _enhanced_preprocess_gray(frame)
                
                # Detect face
                face_box = None
                for params in detection_params:
                    faces = face_cascade.detectMultiScale(enhanced_gray, **params)
                    if len(faces) > 0:
                        face_box = max(faces, key=lambda b: b[2] * b[3])
                        break
                
                # Predict emotion if face detected
                if face_box is not None:
                    x, y, w, h = face_box
                    quality_score, is_acceptable = _assess_frame_quality(enhanced_gray, face_box)
                    
                    if is_acceptable:
                        roi_gray = enhanced_gray[y:y+h, x:x+w]
                        try:
                            roi_preprocessed = _preprocess_roi_for_model(roi_gray)
                            pred = face_model.predict(roi_preprocessed, verbose=0)[0]
                            idx_pred = int(np.argmax(pred))
                            label = labels.get(idx_pred, "neutral")
                            conf = float(pred[idx_pred])
                            
                            # Get all emotion confidences
                            happy_conf = float(pred[3])
                            sad_conf = float(pred[5])
                            surprise_conf = float(pred[6])
                            neutral_conf = float(pred[4])
                            
                            # Minimum confidence threshold
                            MIN_CONFIDENCE_THRESHOLD = 0.4
                            if conf < MIN_CONFIDENCE_THRESHOLD:
                                label = 'neutral'
                                conf = neutral_conf if neutral_conf > 0.2 else conf
                            
                            # Special handling for surprised - require higher confidence
                            if label == 'surprise':
                                if conf < 0.5:
                                    if neutral_conf > 0.3 and abs(surprise_conf - neutral_conf) < 0.15:
                                        label = 'neutral'
                                        conf = neutral_conf
                                    elif happy_conf > 0.3 and abs(surprise_conf - happy_conf) < 0.15:
                                        label = 'happy'
                                        conf = happy_conf
                                    else:
                                        label = 'neutral'
                                        conf = neutral_conf if neutral_conf > 0.2 else conf
                            
                            # Special handling: if happy and sad are close, prefer happy
                            if abs(happy_conf - sad_conf) < 0.15 and happy_conf > 0.3:
                                label = 'happy'
                                conf = happy_conf
                            
                            # Apply temporal smoothing
                            smoothed_label, smoothed_conf = _apply_temporal_smoothing(label, conf)
                            
                            # Final safeguard: if smoothed result is surprise with low confidence, default to neutral
                            if smoothed_label == 'surprise' and smoothed_conf < 0.5:
                                smoothed_label = 'neutral'
                                smoothed_conf = max(smoothed_conf, 0.3)
                            
                            # Update shared state
                            with _face_lock:
                                latest_face_emotion.update({
                                    "label": smoothed_label,
                                    "confidence": round(smoothed_conf, 3),
                                    "ts": datetime.utcnow().isoformat() + "Z"
                                })
                            
                            # Draw overlay
                            color = (0, 255, 0) if smoothed_conf > 0.5 else (0, 165, 255)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            label_text = f"{smoothed_label.upper()}: {smoothed_conf:.2f}"
                            cv2.putText(frame, label_text, (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        except Exception:
                            pass
                    else:
                        # Draw face box even if quality is low
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                        cv2.putText(frame, "Low quality", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                else:
                    # No face detected
                    cv2.putText(frame, "No face detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Encode frame as JPEG
            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception:
                break
        
        # Cleanup
        with _video_cap_lock:
            if _video_cap is not None:
                _video_cap.release()
                _video_cap = None
            _video_running = False
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------------
# Wellness helpers
# ------------------------------
def ambient_risk_from_json(sound_db=None, light=None, temperature=None):
    risks = []
    if sound_db is not None:
        try:
            s = float(sound_db)
            if s < 40:   risks.append(0.2)
            elif s < 60: risks.append(0.4)
            elif s < 80: risks.append(0.6)
            else:        risks.append(0.8)
        except:
            pass
    if light is not None:
        try:
            l = int(light)
            risks.append(0.7 if l == 0 else 0.3)
        except:
            pass
    if temperature is not None:
        try:
            t = float(temperature)
            if t < 20:   risks.append(0.6)
            elif t > 30: risks.append(0.7)
            else:        risks.append(0.4)
        except:
            pass
    if not risks:
        return 0.4
    return float(np.mean(risks))

def physiological_risk(heart_rate=None):
    if heart_rate is None:
        return 0.4
    try:
        hr = float(heart_rate)
        if hr < 60:    return 0.3
        elif hr < 80:  return 0.4
        elif hr < 100: return 0.6
        else:          return 0.8
    except:
        return 0.4

def face_emotion_risk(face_label):
    if not face_label:
        return 0.45
    key = str(face_label).lower()
    return EMOTION_TO_RISK_NUM.get(key, 0.5)

def compute_wellness_index_from_components(audio_risk, env_risk, phys_risk, face_risk):
    total = float(np.clip(W_AUDIO*audio_risk + W_ENV*env_risk + W_PHYS*phys_risk + W_FACE*face_risk, 0.0, 1.0))
    wellness_index = int(round(100 * (1.0 - total)))
    return wellness_index, total

def recommendation_from_wellness(wellness):
    if wellness >= 70:
        return "You’re doing great! Hydrate and maintain good posture."
    elif wellness >= 40:
        return "Moderate fatigue/stress. Try 2-min deep breathing or a short walk."
    else:
        return "High fatigue/stress detected. Please take a 5-minute break and rest."

# ------------------------------
# User & DB helpers (unchanged)
# ------------------------------
def get_user_id_from_request(request):
    user_id = request.cookies.get('user_id')
    if user_id and user_id.isdigit():
        return int(user_id)
    return None

def save_emotion_record(user_id, emotion_data):
    if not user_id:
        return None
    try:
        conn = sqlite3.connect('emotion_analyzer.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO emotion_records 
        (user_id, emotion, confidence, stress_level, recommendation, audio_duration)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            emotion_data['emotion'],
            emotion_data['confidence'],
            emotion_data['stress_level'],
            emotion_data['recommendation'],
            emotion_data.get('audio_duration', 0.0)
        ))
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return record_id
    except Exception as e:
        logger.error(f"Error saving emotion record: {e}")
        return None

# ------------------------------
# Routes: index, analyze
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 204

    temp_file_path = None
    try:
        logger.info(f"Received analyze request with content type: {request.content_type}")

        # Accept audio in file/base64/raw as before
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No selected file'})
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file_path = temp_file.name
                audio_file.save(temp_file_path)
        elif 'audio_data' in request.form:
            audio_data = request.form['audio_data']
            if 'base64,' in audio_data:
                audio_data = audio_data.split('base64,')[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file_path = temp_file.name
                audio_bytes = base64.b64decode(audio_data)
                temp_file.write(audio_bytes)
        elif request.data:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(request.data)
        else:
            return jsonify({'error': 'No audio data provided in any supported format'})

        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) < 100:
            return jsonify({'error': 'Invalid or empty audio file'})

        # predict audio emotion
        audio_result = predict_emotion(temp_file_path)
        logger.info(f"Audio emotion: {audio_result.get('emotion')} ({audio_result.get('confidence_str')})")

        # sensors parsing fallback to live file
        sensor_payload = {}
        if 'sensor_json' in request.form:
            try:
                sensor_payload = json.loads(request.form.get('sensor_json') or "{}")
            except Exception as e:
                logger.warning(f"Failed to parse sensor_json: {e}")
        elif request.is_json:
            try:
                sensor_payload = request.get_json(silent=True) or {}
            except Exception as e:
                logger.warning(f"Failed to parse JSON body: {e}")
        if not sensor_payload:
            sensor_payload = load_live_sensors()

        heart_rate  = sensor_payload.get("heart_rate")
        temperature = sensor_payload.get("temperature")
        light       = sensor_payload.get("light")
        sound_db    = sensor_payload.get("sound_db")

        # audio risk
        emo_key = str(audio_result['emotion']).lower().strip()
        audio_risk = EMOTION_TO_RISK_NUM.get(emo_key, 0.5)

        env_risk  = ambient_risk_from_json(sound_db, light, temperature)
        phys_risk = physiological_risk(heart_rate)

        # face emotion risk (use latest_face_emotion)
        with _face_lock:
            face_label = latest_face_emotion.get("label")
            face_conf  = latest_face_emotion.get("confidence", 0.0)
        face_risk = face_emotion_risk(face_label)

        wellness_index, total_risk = compute_wellness_index_from_components(audio_risk, env_risk, phys_risk, face_risk)
        wellness_tip = recommendation_from_wellness(wellness_index)

        audio_result.update({
            "sensors": {
                "heart_rate": heart_rate,
                "temperature": temperature,
                "light": light,
                "sound_db": sound_db
            },
            "face_emotion": {"label": face_label, "confidence": face_conf},
            "audio_risk": round(audio_risk, 3),
            "env_risk": round(env_risk, 3),
            "phys_risk": round(phys_risk, 3),
            "face_risk": round(face_risk, 3),
            "risk": round(total_risk, 3),
            "wellness_index": wellness_index,
            "wellness_recommendation": wellness_tip
        })

        # save record if logged in
        user_id = get_user_id_from_request(request)
        if user_id:
            record_id = save_emotion_record(user_id, audio_result)
            audio_result['record_id'] = record_id

        return jsonify(audio_result)

    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        traceback.print_exc()

        # fallback logic similar to your existing fallback (preserve wellness computation)
        try:
            emotions = list(id2label.values())
            emotion = random.choice(emotions)
            confidence = random.uniform(0.7, 0.9)
            stress_level = EMOTION_TO_STRESS.get(emotion, 'medium')
            recommendation = random.choice(EMOTION_RECOMMENDATIONS.get(emotion, EMOTION_RECOMMENDATIONS['neutral']))
            all_probs = {e: random.uniform(0.01, 0.2) for e in emotions}
            all_probs[emotion] = confidence

            # sensors fallback
            sensor_payload = {}
            if 'sensor_json' in request.form:
                try:
                    sensor_payload = json.loads(request.form.get('sensor_json') or "{}")
                except:
                    sensor_payload = {}
            elif request.is_json:
                sensor_payload = request.get_json(silent=True) or {}
            if not sensor_payload:
                sensor_payload = load_live_sensors()

            heart_rate  = sensor_payload.get("heart_rate")
            temperature = sensor_payload.get("temperature")
            light       = sensor_payload.get("light")
            sound_db    = sensor_payload.get("sound_db")

            audio_risk = EMOTION_TO_RISK_NUM.get(emotion.lower(), 0.5)
            env_risk  = ambient_risk_from_json(sound_db, light, temperature)
            phys_risk = physiological_risk(heart_rate)
            with _face_lock:
                face_label = latest_face_emotion.get("label")
                face_conf  = latest_face_emotion.get("confidence", 0.0)
            face_risk = face_emotion_risk(face_label)
            wellness_index, total_risk = compute_wellness_index_from_components(audio_risk, env_risk, phys_risk, face_risk)
            wellness_tip = recommendation_from_wellness(wellness_index)

            return jsonify({
                'emotion': emotion,
                'confidence': confidence,
                'confidence_str': f"{confidence:.2%}",
                'stress_level': stress_level,
                'recommendation': recommendation,
                'all_probabilities': all_probs,
                'audio_duration': 5.0,
                'note': 'Using emergency fallback mode due to error',
                "sensors": {
                    "heart_rate": heart_rate,
                    "temperature": temperature,
                    "light": light,
                    "sound_db": sound_db
                },
                "face_emotion": {"label": face_label, "confidence": face_conf},
                "audio_risk": round(audio_risk, 3),
                "env_risk": round(env_risk, 3),
                "phys_risk": round(phys_risk, 3),
                "face_risk": round(face_risk, 3),
                "risk": round(total_risk, 3),
                "wellness_index": wellness_index,
                "wellness_recommendation": wellness_tip
            })
        except Exception as err:
            logger.error(f"Emergency fallback failed: {err}")
            return jsonify({'error': str(e)})
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

# ------------------------------
# Simple wellness endpoint using live sensors + face emotion
# ------------------------------
@app.route('/wellness', methods=['GET'])
def get_wellness():
    sensor = load_live_sensors()
    with _face_lock:
        face_label = latest_face_emotion.get("label")
        face_conf  = latest_face_emotion.get("confidence", 0.0)
    # Use a default audio_risk of 0.5 here (audio not included)
    audio_risk = 0.5
    env_risk = ambient_risk_from_json(sensor.get("sound_db"), sensor.get("light"), sensor.get("temperature"))
    phys_risk = physiological_risk(sensor.get("heart_rate"))
    face_risk = face_emotion_risk(face_label)
    wellness_index, total_risk = compute_wellness_index_from_components(audio_risk, env_risk, phys_risk, face_risk)
    return jsonify({
        "sensors": sensor,
        "face_emotion": {"label": face_label, "confidence": face_conf},
        "wellness_index": wellness_index,
        "risk": round(total_risk, 3)
    })

# ------------------------------
# Health check
# ------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'face_model_available': os.path.exists(FACE_MODEL_JSON) and os.path.exists(FACE_MODEL_WEIGHTS)
    })

# ------------------------------
# Run server
# ------------------------------
if __name__ == '__main__':
    logger.info("Starting merged Speech+Face Emotion server...")
    app.run(debug=True, host='0.0.0.0', port=3002)
