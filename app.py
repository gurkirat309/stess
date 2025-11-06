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

from flask import Flask, request, jsonify, render_template, make_response

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

def capture_face_emotion(timeout_seconds=3):
    """
    Open camera briefly, capture one frame, run face detection/prediction.
    Returns dict: {label, confidence, ts} and updates latest_face_emotion under lock.
    This function is synchronous and intended to be called on-demand (not in a persistent thread).
    """
    face_model = safe_load_face_model()
    if face_model is None:
        return {"label": None, "confidence": 0.0, "ts": None, "error": "face model not available"}

    # try to open camera (use CAP_DSHOW on Windows if needed)
    try:
        # Note: on macOS/OpenCV, sometimes CAP_DSHOW isn't available — keep default
        cap = cv2.VideoCapture(CAMERA_INDEX)
        start = time.time()
        if not cap.isOpened():
            # try fallback backends
            try:
                cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            except Exception:
                pass

        # warm-up for a short time
        warm_started = time.time()
        while time.time() - warm_started < 0.3:
            if cap.isOpened():
                break
            time.sleep(0.05)

        if not cap.isOpened():
            return {"label": None, "confidence": 0.0, "ts": None, "error": "camera not available"}

        # set a reasonable resolution
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except:
            pass

        # read one good frame within timeout
        frame = None
        while time.time() - start < timeout_seconds:
            ret, f = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = f
            break

        cap.release()

        if frame is None:
            return {"label": None, "confidence": 0.0, "ts": None, "error": "no frame captured"}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return {"label": None, "confidence": 0.0, "ts": None, "error": "no face detected"}

        # pick largest face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        pred = face_model.predict(roi, verbose=0)[0]
        idx = int(np.argmax(pred))
        # map to the same labels you used in training
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        label = labels.get(idx, "neutral")
        conf = float(pred[idx])

        out = {"label": label, "confidence": round(conf, 3), "ts": datetime.utcnow().isoformat() + "Z"}

        # update shared state
        with _face_lock:
            latest_face_emotion.update({"label": out["label"], "confidence": out["confidence"], "ts": out["ts"]})

        return out

    except Exception as e:
        logger.exception(f"capture_face_emotion error: {e}")
        try:
            if 'cap' in locals() and cap and cap.isOpened():
                cap.release()
        except:
            pass
        return {"label": None, "confidence": 0.0, "ts": None, "error": str(e)}

# ------------------------------
# Routes for face emotion
# ------------------------------
@app.route('/capture-face', methods=['GET'])
def capture_face_route():
    """
    Capture one frame from the camera, run face emotion detection, return result.
    This is safe to call from UI or a client. It does not start a persistent thread.
    """
    result = capture_face_emotion()
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
    app.run(debug=True, host='0.0.0.0', port=5000)
