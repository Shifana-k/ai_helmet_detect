import cv2
from deepface import DeepFace
import pickle
import numpy as np
import os

def capture_face_encodings():
    encodings = {}
    
    # Load existing encodings if available
    if os.path.exists("face_encodings.pkl"):
        with open("face_encodings.pkl", "rb") as f:
            encodings = pickle.load(f)
        print("Loaded existing encodings:")
        for name in encodings.keys():
            print(f"  - {name}")
        print()
    
    cap = cv2.VideoCapture(0)
    
    print("=" * 60)
    print("FACE CAPTURE TOOL")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Position your face clearly in the camera")
    print("2. Press SPACE to capture")
    print("3. Type your name when prompted")
    print("4. Capture multiple angles for better accuracy!")
    print("5. Press 'q' to finish and save")
    print("\nTips:")
    print("- Good lighting helps!")
    print("- Face the camera directly")
    print("- Capture 2-3 photos per person from different angles")
    print("=" * 60)
    
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show instructions on frame
        cv2.putText(frame, "Press SPACE to capture face", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press Q to quit and save", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Capture Faces", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar
            print(f"\n📸 Capturing face #{capture_count + 1}...")
            
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = DeepFace.represent(rgb_frame, model_name="Facenet512", enforce_detection=True)
                
                if result:
                    embedding = np.array(result[0]["embedding"])
                    
                    # Get name from user
                    name = input("Enter name for this face: ").strip()
                    
                    if name:
                        # If name exists, average with existing encoding
                        if name in encodings:
                            print(f"Adding another sample for {name}...")
                            existing = np.array(encodings[name])
                            # Average multiple captures for better accuracy
                            encodings[name] = ((existing + embedding) / 2).tolist()
                        else:
                            encodings[name] = embedding.tolist()
                        
                        capture_count += 1
                        print(f"✅ Captured {name}! Total captures: {capture_count}")
                        print(f"Current people: {list(encodings.keys())}")
                    else:
                        print("❌ No name entered, skipping...")
                else:
                    print("❌ No face detected! Try again with better lighting.")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Make sure your face is clearly visible and well-lit!")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if encodings:
        with open("face_encodings.pkl", "wb") as f:
            pickle.dump(encodings, f)
        
        print("\n" + "=" * 60)
        print("✅ FACE ENCODINGS SAVED!")
        print("=" * 60)
        print(f"Total people encoded: {len(encodings)}")
        for name in encodings.keys():
            print(f"  ✓ {name}")
        print("\nYou can now run main.py!")
        print("=" * 60)
    else:
        print("\n❌ No faces captured!")

if __name__ == "__main__":
    capture_face_encodings()