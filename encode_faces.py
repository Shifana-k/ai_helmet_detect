import os
import cv2
from deepface import DeepFace
import pickle
import numpy as np

# Folder containing  family photos
FAMILY_FOLDER = "family_faces"

encodings = {}

print("Encoding faces...")

for filename in os.listdir(FAMILY_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(FAMILY_FOLDER, filename)

        print(f"Encoding {name}...")

        try:
            embedding = DeepFace.represent(
                img_path,
                model_name="Facenet512",
                enforce_detection=False
            )

            encodings[name] = np.array(embedding[0]["embedding"])
        except Exception as e:
            print(f"Error encoding {name}: {e}")

# Save encodings
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(encodings, f)

print("Encoding completed! Saved as face_encodings.pkl")
