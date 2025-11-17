import cv2
from ultralytics import YOLO
from deepface import DeepFace
import pickle
from gtts import gTTS
import os
from playsound import playsound
import numpy as np
import time
import threading

# Load YOLO model
helmet_model = YOLO("helmet_yolo/helmet.pt")

# Load face encodings
with open("face_encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Track last alert to avoid spam
last_alert_time = {}
ALERT_COOLDOWN = 5  # seconds

def speak(text, use_malayalam=False):
    """Speak text using gTTS in a separate thread"""
    def _speak():
        try:
            if use_malayalam:
                lang = "ml"
            else:
                lang = "en"
            
            tts = gTTS(text=text, lang=lang)
            filename = f"alert_{time.time()}.mp3"
            tts.save(filename)
            playsound(filename)
            time.sleep(0.5)
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f"TTS error: {e}")
    
    # Run in separate thread so it doesn't block video
    threading.Thread(target=_speak, daemon=True).start()

def recognize_face(frame):
    """Recognize face using DeepFace"""
    try:
        result = DeepFace.represent(frame, model_name="Facenet512", enforce_detection=False)
        
        if not result:
            return None
            
        embedding = np.array(result[0]["embedding"])
        
        # Find closest match
        best_match = None
        min_distance = float('inf')
        
        for name, saved_embedding in known_faces.items():
            saved_embedding = np.array(saved_embedding)
            distance = np.linalg.norm(saved_embedding - embedding)
            
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        # Threshold for recognition (increased for better matching)
        if min_distance < 10.0:  # Adjusted based on your test results
            return best_match
            
    except Exception as e:
        print(f"Face recognition error: {e}")
    
    return None

def should_alert(person_id):
    """Check if enough time has passed since last alert"""
    current_time = time.time()
    if person_id not in last_alert_time:
        last_alert_time[person_id] = current_time
        return True
    
    if current_time - last_alert_time[person_id] > ALERT_COOLDOWN:
        last_alert_time[person_id] = current_time
        return True
    
    return False

# Initialize camera
cap = cv2.VideoCapture(0)

frame_counter = 0
FACE_CHECK_INTERVAL = 10  # Check face every 10 frames (for performance)
current_person = None

print("Starting AI Helmet Camera...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_counter += 1
    
    # Helmet detection (every frame)
    results = helmet_model(frame, verbose=False)
    helmet_detected = False
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Helmet class
                helmet_detected = True
                break
    
    # Face recognition (every N frames to save processing)
    if frame_counter % FACE_CHECK_INTERVAL == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_person = recognize_face(rgb_frame)
    
    # Determine display and alert
    if current_person:
        display_name = current_person
        person_id = current_person
    else:
        display_name = "Unknown"
        person_id = "unknown"
    
    # Alert logic
    if not helmet_detected:
        if should_alert(person_id):
            if current_person:
                # Malayalam message
                message = f"{current_person}, ഹെൽമെറ്റ് എവിടെ?"  # "helmet evide?"
                message_en = f"{current_person}, where is your helmet?"
            else:
                message = "ഹേയ്! ഹെൽമെറ്റ് എവിടെ?"  # "Hey! helmet evide?"
                message_en = "Hey! Where is your helmet?"
            
            print(message_en)
            speak(message, use_malayalam=True)  # Use Malayalam
        
        # Display
        color = (0, 0, 255)  # Red
        status = "NO HELMET"
    else:
        if should_alert(person_id):
            if current_person:
                message = f"{current_person}, ഹെൽമെറ്റ് ശരിയാണ്! നന്നായി!"  # "helmet shariyaan! nannaayi!"
                message_en = f"{current_person}, helmet OK! Good job!"
            else:
                message = "ഹെൽമെറ്റ് ശരിയാണ്!"  # "helmet shariyaan!"
                message_en = "Helmet OK!"
            
            print(message_en)
            speak(message, use_malayalam=True)  # Use Malayalam
        
        # Display
        color = (0, 255, 0)  # Green
        status = "Helmet OK"
    
    # Draw on frame
    cv2.putText(frame, display_name, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, status, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Show helmet bounding boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    cv2.imshow("AI Helmet Camera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")