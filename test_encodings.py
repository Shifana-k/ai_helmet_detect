import pickle
import cv2
from deepface import DeepFace
import numpy as np

# Load your saved encodings
with open("face_encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)

print("=" * 50)
print("SAVED FACE ENCODINGS:")
print("=" * 50)
for name, encoding in known_faces.items():
    print(f"✓ {name}: {len(encoding)} dimensions")
print()

# Now test with live camera
print("=" * 50)
print("TESTING LIVE RECOGNITION")
print("=" * 50)
print("Opening camera... Press SPACE to test recognition")
print("Press 'q' to quit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Test - Press SPACE to check face", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):  # Space bar pressed
        print("\n🔍 Analyzing face...")
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = DeepFace.represent(rgb_frame, model_name="Facenet512", enforce_detection=False)
            
            if not result:
                print("❌ No face detected in frame!")
                continue
            
            embedding = np.array(result[0]["embedding"])
            print(f"✓ Face detected! Embedding: {len(embedding)} dimensions")
            
            print("\nComparing with saved faces:")
            print("-" * 50)
            
            best_match = None
            min_distance = float('inf')
            
            for name, saved_embedding in known_faces.items():
                saved_embedding = np.array(saved_embedding)
                distance = np.linalg.norm(saved_embedding - embedding)
                
                print(f"{name:15} - Distance: {distance:.4f}")
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = name
            
            print("-" * 50)
            print(f"\n🎯 Best match: {best_match} (distance: {min_distance:.4f})")
            
            if min_distance < 0.6:
                print(f"✅ RECOGNIZED as {best_match}")
            else:
                print(f"❌ NOT RECOGNIZED (threshold: 0.6)")
                print(f"💡 Try increasing threshold to {min_distance + 0.1:.2f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()