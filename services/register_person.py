import os
import argparse
import numpy as np
from insightface.app import FaceAnalysis
from database import DatabaseService
import cv2
import uuid

def register_people(image_dir: str):
    """Registers multiple people from a directory of images."""

    # Initialize InsightFace
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    db = DatabaseService()

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)

            # Load image and extract embedding
            img = cv2.imread(image_path)
            faces = app.get(img)
            if not faces:
                print(f"No face detected in {filename}. Skipping.")
                continue

            embedding = faces[0].embedding

            # Use filename as person name (without extension)
            name = os.path.splitext(filename)[0]

            # Generate unique ID
            person_id = str(uuid.uuid4())

            # Store in database
            db.store_person_embedding(person_id, embedding, name)
            print(f"Person {name} (ID: {person_id}) registered successfully from {filename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register people from a directory of images.")
    parser.add_argument("--image_dir", default="reg_imgs", type=str, help="Path to the directory containing images.")
    args = parser.parse_args()

    register_people(args.image_dir)