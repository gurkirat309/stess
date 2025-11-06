from flask import Flask, request, jsonify, render_template, make_response
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

# ------------------------------
# --- Wellness additions (config)
# ------------------------------
# Combine risks (tweak later if needed)
W_AUDIO = 0.5   # emotion-derived stress weight
W_ENV   = 0.3   # environment (sound/light/temp)
W_PHYS  = 0.2   # physiology (heart rate)

# Map emotion -> base stress risk [0..1]
EMOTION_TO_RISK_NUM = {
    'angry': 0.90,
    'disgust': 0.80,
    'fearful': 0.85,
    'sad': 0.60,
    'surprised': 0.55,
    'neutral': 0.45,
    'happy': 0.25,
    # 'calm': 0.15,  # only if present in your labels
}

def ambient_risk_from_json(sound_db=None, light=None, temperature=None):
    """Heuristic env risk (0..1) from provided sensors.
       light: 0=dark, 1=light
    """
    risks = []
    # Sound (dB)
    if sound_db is not None:
        try:
            s = float(sound_db)
            if s < 40:   risks.append(0.2)
            elif s < 60: risks.append(0.4)
            elif s < 80: risks.append(0.6)
            else:        risks.append(0.8)
        except:  # ignore parse errors
            pass
    # Light (binary)
    if light is not None:
        try:
            l = int(light)
            risks.append(0.7 if l == 0 else 0.3)
        except:
            pass
    # Temperature (°C)
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
    """HR -> stress risk (0..1). Simple bins you can refine later."""
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

def recommendation_from_wellness(wellness):
    if wellness >= 70:
        return "You’re doing great! Hydrate and maintain good posture."
    elif wellness >= 40:
        return "Moderate fatigue/stress. Try 2-min deep breathing or a short walk."
    else:
        return "High fatigue/stress detected. Please take a 5-minute break and rest."

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder=".")
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# Add CORS headers to support different clients
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize database
def init_db():
    conn = sqlite3.connect('emotion_analyzer.db')
    cursor = conn.cursor()
    
    # Create users table
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
    
    # Create emotion_records table
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

# Load the model and feature extractor
print("Loading model...")
try:
    model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
    id2label = model.config.id2label
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using fallback mode")
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

# Define emotion to recommendations mapping
EMOTION_RECOMMENDATIONS = {
    'angry': [
        "Take deep breaths and count to 10",
        "Go for a short walk to cool down",
        "Listen to calming music",
        "Practice progressive muscle relaxation",
        "Write down what's making you angry"
    ],
    'disgust': [
        "Focus on positive aspects of your environment",
        "Practice mindfulness to center yourself",
        "Move to a different location if possible",
        "Engage in a pleasant sensory experience",
        "Talk to someone about your feelings"
    ],
    'fearful': [
        "Use grounding techniques (name 5 things you can see, 4 things you can touch, etc.)",
        "Remember that feelings pass with time",
        "Practice deep breathing exercises",
        "Challenge irrational thoughts",
        "Reach out to a supportive friend"
    ],
    'happy': [
        "Savor this positive emotion fully",
        "Share your happiness with others",
        "Express gratitude for what's going well",
        "Engage in activities you enjoy",
        "Document this moment to revisit later"
    ],
    'neutral': [
        "Reflect on your current goals and priorities",
        "Use this balanced state for decision-making",
        "Practice gratitude for your emotional balance",
        "Set intentions for the rest of your day",
        "Check in with your physical needs (hunger, thirst, rest)"
    ],
    'sad': [
        "Allow yourself to feel without judgment",
        "Engage in gentle physical activity",
        "Connect with a friend or family member",
        "Practice self-compassion",
        "Do something creative or expressive"
    ],
    'surprised': [
        "Take a moment to process the unexpected event",
        "Focus on your breathing to center yourself",
        "Assess whether any action is needed",
        "Share your experience with someone",
        "Use the energy of surprise for creativity"
    ]
}

# Define emotion to stress level mapping
EMOTION_TO_STRESS = {
    'angry': 'high',
    'disgust': 'high',
    'fearful': 'high',
    'sad': 'high',
    'surprised': 'medium',
    'neutral': 'medium',
    'happy': 'low'
}

def load_audio(audio_path):
    """Load audio file with multiple fallback methods"""
    try:
        # Primary method: librosa
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        logger.info(f"Loaded audio with librosa: length={len(audio_array)}, sr={sampling_rate}")
        return audio_array, sampling_rate
    except Exception as e:
        logger.warning(f"Librosa failed to load audio: {e}")
        try:
            # Try with soundfile
            import soundfile as sf
            audio_array, sampling_rate = sf.read(audio_path)
            # Convert to mono if stereo
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
            # Resample to 16kHz if needed
            if sampling_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
                sampling_rate = 16000
            logger.info(f"Loaded audio with soundfile: length={len(audio_array)}, sr={sampling_rate}")
            return audio_array, sampling_rate
        except Exception as sf_e:
            logger.warning(f"SoundFile failed to load audio: {sf_e}")
            # Last resort: wave module
            try:
                import wave
                with wave.open(audio_path, 'rb') as wf:
                    frames = wf.getnframes()
                    sample_rate = wf.getframerate()
                    sample_width = wf.getsampwidth()
                    channels = wf.getnchannels()
                    raw_data = wf.readframes(frames)
                    
                    # Convert to float32 normalized array
                    if sample_width == 2:
                        data = np.frombuffer(raw_data, dtype=np.int16)
                        audio_array = data.astype(np.float32) / 32768.0
                    else:
                        data = np.frombuffer(raw_data, dtype=np.int8)
                        audio_array = data.astype(np.float32) / 128.0
                    
                    # Convert to mono if stereo
                    if channels > 1:
                        audio_array = audio_array.reshape(-1, channels).mean(axis=1)
                    
                    # Resample if needed
                    if sample_rate != 16000:
                        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                        sample_rate = 16000
                    
                    logger.info(f"Loaded audio with wave: length={len(audio_array)}, sr={sample_rate}")
                    return audio_array, sample_rate
            except Exception as wave_e:
                logger.error(f"All audio loading methods failed: {wave_e}")
                raise Exception("Failed to load audio file with any available method")

def preprocess_audio(audio_path, max_duration=30.0):
    """Preprocess audio for the Whisper model"""
    try:
        # Load audio file
        audio_array, sampling_rate = load_audio(audio_path)
        
        # Handle maximum duration
        max_length = int(16000 * max_duration)  # 16kHz sample rate
        if len(audio_array) > max_length:
            logger.info(f"Audio too long ({len(audio_array)} samples), truncating to {max_length} samples")
            audio_array = audio_array[:max_length]
        
        # Handle very short recordings
        if len(audio_array) < 8000:  # Less than 0.5 seconds
            logger.info(f"Audio too short ({len(audio_array)} samples), padding")
            audio_array = np.pad(audio_array, (0, 8000 - len(audio_array)), 'constant')
        
        # Normalize audio
        audio_array = librosa.util.normalize(audio_array)
        
        # Extract features
        if feature_extractor:
            inputs = feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            )
            return inputs, audio_array
        else:
            # Fallback for testing when model isn't loaded
            return None, audio_array
            
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        traceback.print_exc()
        raise

def predict_emotion(audio_path, max_duration=30.0):
    """Predict emotion from audio file"""
    try:
        # Fallback mode if model not loaded
        if model is None:
            # Generate random probabilities for testing
            emotions = list(id2label.values())
            probs = {emotion: random.uniform(0.01, 0.2) for emotion in emotions}
            emotion = random.choice(emotions)
            probs[emotion] = random.uniform(0.6, 0.9)  # Give the "detected" emotion a high probability
            stress_level = EMOTION_TO_STRESS.get(emotion, 'medium')
            recommendation = random.choice(EMOTION_RECOMMENDATIONS.get(emotion, EMOTION_RECOMMENDATIONS['neutral']))
            
            logger.info(f"Using fallback mode. Selected emotion: {emotion}")
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
        
        # Preprocess audio
        inputs, audio_array = preprocess_audio(audio_path, max_duration)
        
        # Setup device (CPU/GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_id = torch.argmax(logits, dim=-1).item()
            predicted_label = id2label[predicted_id]
            confidence = probabilities[0][predicted_id].item()
            
            # Get all probabilities
            all_probs = {id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])}
            
        # Map to stress level
        stress_level = EMOTION_TO_STRESS.get(predicted_label, 'medium')
        
        # Get recommendation
        recommendations = EMOTION_RECOMMENDATIONS.get(predicted_label, EMOTION_RECOMMENDATIONS['neutral'])
        recommendation = random.choice(recommendations)
        
        logger.info(f"Prediction complete. Emotion: {predicted_label}, Confidence: {confidence:.2%}")
        
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

def get_user_id_from_request(request):
    """Extract user ID from cookies"""
    user_id = request.cookies.get('user_id')
    if user_id and user_id.isdigit():
        return int(user_id)
    return None

def save_emotion_record(user_id, emotion_data):
    """Save emotion analysis record to database"""
    if not user_id:
        logger.info("No user ID provided, skipping record save")
        return
    
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
            emotion_data['audio_duration']
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Saved emotion record {record_id} for user {user_id}")
        return record_id
    except Exception as e:
        logger.error(f"Error saving emotion record: {e}")
        return None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Analyze uploaded or recorded audio"""
    # Handle preflight requests for CORS
    if request.method == 'OPTIONS':
        return '', 204
        
    temp_file_path = None
    try:
        logger.info(f"Received analyze request with content type: {request.content_type}")
        
        # Check if request contains an uploaded file
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No selected file'})
            
            logger.info(f"Processing uploaded file: {audio_file.filename}")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file_path = temp_file.name
                audio_file.save(temp_file_path)
                logger.info(f"Saved uploaded file to: {temp_file_path}")
                
        # Check if request contains base64 audio data (from live recording)
        elif 'audio_data' in request.form:
            audio_data = request.form['audio_data']
            logger.info("Received base64 audio data")
            
            # Remove the data URL prefix if present
            if 'base64,' in audio_data:
                audio_data = audio_data.split('base64,')[1]
                logger.info("Extracted base64 data from data URL")
            
            # Decode base64 audio data
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_file_path = temp_file.name
                    audio_bytes = base64.b64decode(audio_data)
                    temp_file.write(audio_bytes)
                    logger.info(f"Decoded base64 audio to: {temp_file_path} ({len(audio_bytes)} bytes)")
            except Exception as e:
                logger.error(f"Error decoding base64: {e}")
                return jsonify({'error': f'Failed to decode audio data: {str(e)}'})
                
        # Handle raw binary data
        elif request.data:
            logger.info(f"Received raw audio data ({len(request.data)} bytes)")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(request.data)
                logger.info(f"Saved raw audio data to: {temp_file_path}")
        else:
            logger.error("No audio data found in request")
            return jsonify({'error': 'No audio data provided in any supported format'})
        
        # Check if the file exists and has content
        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) < 100:
            logger.error(f"Invalid audio file: {temp_file_path}, size: {os.path.getsize(temp_file_path) if os.path.exists(temp_file_path) else 'file not found'}")
            return jsonify({'error': 'Invalid or empty audio file'})
        
        # Predict emotion
        result = predict_emotion(temp_file_path)
        logger.info(f"Analysis complete: {result['emotion']} ({result['confidence_str']})")

        # ------------------------------
        # --- Wellness additions (compute from JSON sensors)
        # Supported sources:
        # 1) multipart/form-data field: sensor_json='{"heart_rate":..,"temperature":..,"light":0/1,"sound_db":..}'
        # 2) raw JSON body (when client posts JSON along with/without audio)
        # ------------------------------
        sensor_payload = {}
        # (a) multipart form field
        if 'sensor_json' in request.form:
            try:
                sensor_payload = json.loads(request.form.get('sensor_json') or "{}")
            except Exception as e:
                logger.warning(f"Failed to parse sensor_json field: {e}")
        # (b) raw JSON body
        elif request.is_json:
            try:
                sensor_payload = request.get_json(silent=True) or {}
            except Exception as e:
                logger.warning(f"Failed to read JSON body: {e}")

        heart_rate  = sensor_payload.get("heart_rate")
        temperature = sensor_payload.get("temperature")
        light       = sensor_payload.get("light")       # 0 or 1
        sound_db    = sensor_payload.get("sound_db")    # dB

        # Audio (emotion) risk
        emo_key = str(result['emotion']).lower().strip()
        audio_risk = EMOTION_TO_RISK_NUM.get(emo_key, 0.5)

        # Environment + Physiology risks
        env_risk  = ambient_risk_from_json(sound_db, light, temperature)
        phys_risk = physiological_risk(heart_rate)

        total_risk = float(np.clip(W_AUDIO*audio_risk + W_ENV*env_risk + W_PHYS*phys_risk, 0.0, 1.0))
        wellness_index = int(round(100 * (1.0 - total_risk)))
        wellness_tip = recommendation_from_wellness(wellness_index)

        # Attach to your existing response
        result.update({
            "sensors": {
                "heart_rate": heart_rate,
                "temperature": temperature,
                "light": light,
                "sound_db": sound_db
            },
            "audio_risk": round(audio_risk, 3),
            "env_risk": round(env_risk, 3),
            "phys_risk": round(phys_risk, 3),
            "risk": round(total_risk, 3),
            "wellness_index": wellness_index,
            "wellness_recommendation": wellness_tip
        })
        # --- End wellness additions ---
        
        # Save record if user is logged in (unchanged)
        user_id = get_user_id_from_request(request)
        if user_id:
            record_id = save_emotion_record(user_id, result)
            result['record_id'] = record_id
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        traceback.print_exc()
        
        # Try fallback mode for any errors
        try:
            emotions = list(id2label.values())
            emotion = random.choice(emotions)
            confidence = random.uniform(0.7, 0.9)
            stress_level = EMOTION_TO_STRESS.get(emotion, 'medium')
            recommendation = random.choice(EMOTION_RECOMMENDATIONS.get(emotion, EMOTION_RECOMMENDATIONS['neutral']))
            all_probs = {e: random.uniform(0.01, 0.2) for e in emotions}
            all_probs[emotion] = confidence
            
            logger.info(f"Using emergency fallback due to error. Emotion: {emotion}")
            
            # --- Wellness additions even in fallback (safeguard) ---
            sensor_payload = {}
            if 'sensor_json' in request.form:
                try:
                    sensor_payload = json.loads(request.form.get('sensor_json') or "{}")
                except:
                    sensor_payload = {}
            elif request.is_json:
                sensor_payload = request.get_json(silent=True) or {}

            heart_rate  = sensor_payload.get("heart_rate")
            temperature = sensor_payload.get("temperature")
            light       = sensor_payload.get("light")
            sound_db    = sensor_payload.get("sound_db")

            audio_risk = EMOTION_TO_RISK_NUM.get(emotion.lower(), 0.5)
            env_risk  = ambient_risk_from_json(sound_db, light, temperature)
            phys_risk = physiological_risk(heart_rate)
            total_risk = float(np.clip(W_AUDIO*audio_risk + W_ENV*env_risk + W_PHYS*phys_risk, 0.0, 1.0))
            wellness_index = int(round(100 * (1.0 - total_risk)))
            wellness_tip = recommendation_from_wellness(wellness_index)
            # --- end wellness additions ---

            return jsonify({
                'emotion': emotion,
                'confidence': confidence,
                'confidence_str': f"{confidence:.2%}",
                'stress_level': stress_level,
                'recommendation': recommendation,
                'all_probabilities': all_probs,
                'audio_duration': 5.0,
                'note': 'Using emergency fallback mode due to error',
                # wellness extras
                "sensors": {
                    "heart_rate": heart_rate,
                    "temperature": temperature,
                    "light": light,
                    "sound_db": sound_db
                },
                "audio_risk": round(audio_risk, 3),
                "env_risk": round(env_risk, 3),
                "phys_risk": round(phys_risk, 3),
                "risk": round(total_risk, 3),
                "wellness_index": wellness_index,
                "wellness_recommendation": wellness_tip
            })
        except:
            # If even the fallback fails, return error
            return jsonify({'error': str(e)})
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_e:
                logger.error(f"Error cleaning up temporary file: {cleanup_e}")

@app.route('/register', methods=['POST', 'OPTIONS'])
def register():
    """Register a new user"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')  # In a real app, you'd hash this
        display_name = data.get('display_name', username)
        email = data.get('email', '')
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password are required'})
        
        # Check if username already exists
        conn = sqlite3.connect('emotion_analyzer.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            conn.close()
            return jsonify({'success': False, 'error': 'Username already exists'})
        
        # Create new user
        cursor.execute(
            'INSERT INTO users (username, password, display_name, email) VALUES (?, ?, ?, ?)',
            (username, password, display_name, email)
        )
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Registered new user: {username} (ID: {user_id})")
        
        # Create response with cookie
        response = jsonify({
            'success': True, 
            'user_id': user_id,
            'username': username,
            'display_name': display_name
        })
        
        response.set_cookie('user_id', str(user_id), max_age=30*24*60*60)  # 30 days
        return response
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    """Login a user"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password are required'})
        
        # Check credentials
        conn = sqlite3.connect('emotion_analyzer.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, display_name, password FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if not user or user['password'] != password:  # In a real app, compare hashed passwords
            return jsonify({'success': False, 'error': 'Invalid username or password'})
        
        # Create response with cookie
        response = jsonify({
            'success': True,
            'user_id': user['id'],
            'username': user['username'],
            'display_name': user['display_name']
        })
        
        response.set_cookie('user_id', str(user['id']), max_age=30*24*60*60)  # 30 days
        logger.info(f"User logged in: {username} (ID: {user['id']})")
        return response
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/logout', methods=['POST', 'OPTIONS'])
def logout():
    """Logout a user"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
        
    response = jsonify({'success': True})
    response.delete_cookie('user_id')
    return response

@app.route('/user/profile', methods=['GET'])
def get_profile():
    """Get user profile information"""
    user_id = get_user_id_from_request(request)
    
    if not user_id:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    try:
        conn = sqlite3.connect('emotion_analyzer.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get user info
        cursor.execute('SELECT username, display_name, email, created_at FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'success': False, 'error': 'User not found'})
        
        # Get emotion stats
        cursor.execute('''
        SELECT COUNT(*) as total_records,
               MAX(created_at) as last_record
        FROM emotion_records 
        WHERE user_id = ?
        ''', (user_id,))
        stats = cursor.fetchone()
        
        # Get emotion distribution
        cursor.execute('''
        SELECT emotion, COUNT(*) as count
        FROM emotion_records 
        WHERE user_id = ?
        GROUP BY emotion
        ''', (user_id,))
        emotion_counts = cursor.fetchall()
        
        conn.close()
        
        user_dict = dict(user)
        stats_dict = dict(stats)
        emotion_distribution = {row['emotion']: row['count'] for row in emotion_counts}
        
        return jsonify({
            'success': True,
            'profile': user_dict,
            'stats': stats_dict,
            'emotion_distribution': emotion_distribution
        })
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/user/history', methods=['GET'])
def get_history():
    """Get user emotion analysis history"""
    user_id = get_user_id_from_request(request)
    
    if not user_id:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    try:
        # Get query parameters
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
        
        conn = sqlite3.connect('emotion_analyzer.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get records with pagination
        cursor.execute('''
        SELECT id, emotion, confidence, stress_level, recommendation, 
               audio_duration, created_at
        FROM emotion_records 
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        ''', (user_id, limit, offset))
        
        records = [dict(row) for row in cursor.fetchall()]
        
        # Get total count
        cursor.execute('SELECT COUNT(*) as count FROM emotion_records WHERE user_id = ?', (user_id,))
        total = cursor.fetchone()['count']
        
        conn.close()
        
        return jsonify({
            'success': True,
            'records': records,
            'total': total,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/user/stats', methods=['GET'])
def get_stats():
    """Get user emotion statistics for progress tracking"""
    user_id = get_user_id_from_request(request)
    
    if not user_id:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    try:
        # Get time range
        days = int(request.args.get('days', 30))
        
        conn = sqlite3.connect('emotion_analyzer.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get emotion trends over time
        cursor.execute('''
        SELECT date(created_at) as date, emotion, COUNT(*) as count
        FROM emotion_records 
        WHERE user_id = ? AND created_at >= ?
        GROUP BY date(created_at), emotion
        ORDER BY date(created_at)
        ''', (user_id, start_date.strftime('%Y-%m-%d')))
        
        emotion_trends_raw = cursor.fetchall()
        
        # Get stress level trends
        cursor.execute('''
        SELECT date(created_at) as date, stress_level, COUNT(*) as count
        FROM emotion_records 
        WHERE user_id = ? AND created_at >= ?
        GROUP BY date(created_at), stress_level
        ORDER BY date(created_at)
        ''', (user_id, start_date.strftime('%Y-%m-%d')))
        
        stress_trends_raw = cursor.fetchall()
        
        # Get most common emotions
        cursor.execute('''
        SELECT emotion, COUNT(*) as count
        FROM emotion_records 
        WHERE user_id = ? AND created_at >= ?
        GROUP BY emotion
        ORDER BY count DESC
        ''', (user_id, start_date.strftime('%Y-%m-%d')))
        
        top_emotions = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        # Format trend data for charts
        emotion_trends = {}
        for row in emotion_trends_raw:
            date = row['date']
            emotion = row['emotion']
            count = row['count']
            
            if date not in emotion_trends:
                emotion_trends[date] = {}
                
            emotion_trends[date][emotion] = count
        
        stress_trends = {}
        for row in stress_trends_raw:
            date = row['date']
            level = row['stress_level']
            count = row['count']
            
            if date not in stress_trends:
                stress_trends[date] = {}
                
            stress_trends[date][level] = count
        
        return jsonify({
            'success': True,
            'emotion_trends': emotion_trends,
            'stress_trends': stress_trends,
            'top_emotions': top_emotions,
            'days': days
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'feature_extractor_loaded': feature_extractor is not None,
        'emotions_supported': list(id2label.values()) if id2label else []
    })

if __name__ == '__main__':
    print("Starting Speech Emotion Recognition server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
