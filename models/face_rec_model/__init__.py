from typing import Hashable, Optional
import numpy as np
from models.base_model import BaseModel
import insightface
from insightface.app import FaceAnalysis
import time
from datetime import datetime

from services.database import DatabaseService
class FaceRecognitionModel(BaseModel):
    name = "FaceRecognition"

    def __init__(self, db: Optional[DatabaseService] = None):
        super().__init__()
        # Initialize InsightFace
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.current_face_embedding = None
        self.current_person_id = None
        self.recognition_completed = False
        self.db = db

    def start(self, sid: Hashable, timestamp: int, *args, **kwargs) -> None:
        # print(f"{self.name} started at {datetime.fromtimestamp(timestamp/1000) or "unknown time"} with sid {sid}")
        self.recognition_completed = False
        self.current_face_embedding = None
        self.current_person_id = None
        self.rec_result = None

    def end(self, sid: Hashable, timestamp: Optional[int], *args, **kwargs) -> dict:
        print(f"{self.name} ended at {datetime.fromtimestamp(timestamp/1000) if timestamp else 'unknown time'} with sid {sid}")
        return {"person_id": self.current_person_id}

    def frame(self, sid: Hashable, frame: np.ndarray, timestamp: int, *args, **kwargs) -> Optional[dict]:
        if self.recognition_completed:
            return self.rec_result

        faces = self.app.get(frame)
        if len(faces) == 0:
            return None

        face = faces[0]  # Take the first detected face
        self.current_face_embedding = face.embedding
        
        # Get person_id from database matching
        self.current_person_id = self.db.find_matching_person(self.current_face_embedding)
        self.recognition_completed = True
        self.rec_result = {
            "person_id": self.current_person_id,
            "face_detected": True,
            # "recognition_timestamp": timestamp
        }

        return self.rec_result