from pymongo import MongoClient
from datetime import datetime
import numpy as np
from typing import Optional, List, Dict, Any
import os

class DatabaseService:
    def __init__(self, connection_string: Optional[str] = None):
        if connection_string is None:
            connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        self.client = MongoClient(connection_string)
        self.db = self.client.athlete_monitoring
        
    def store_person_embedding(self, person_id: str, embedding: np.ndarray, name: str):
        self.db.face_embeddings.update_one(
            {"person_id": person_id},
            {"$set": {"embedding": embedding.tolist(), "name": name}},
            upsert=True
        )

    def find_matching_person(self, embedding: np.ndarray, threshold: float = 0.6) -> Optional[str]:
        """Match face embedding with stored embeddings in database"""
        # Get all face embeddings from database
        cursor = self.db.face_embeddings.find({})
        best_match = None
        highest_similarity = -1

        for doc in cursor:
            stored_embedding = np.array(doc['embedding'])
            # Calculate cosine similarity
            similarity = np.dot(embedding, stored_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = doc['person_id']
        
        # Return person_id if similarity exceeds threshold
        if highest_similarity > threshold:
            return best_match
        return None

    def store_measurement(self, person_id: str, measurement_type: str, value: Any, timestamp: int):
        self.db.measurements.insert_one({
            "person_id": person_id,
            "type": measurement_type,
            "value": value,
            "timestamp": timestamp
        })


    def get_person_measurements_summary(self, person_id: str) -> Dict[str, List[Dict]]:
        """Get all measurements for a person grouped by type"""
        measurements = {}
        for measurement_type in ['heart_rate', 'heart_rate_variability', 'fatigue', 'dark_circles', 'pimples']:
            results = list(self.db.measurements.find(
                {"person_id": person_id, "type": measurement_type},
                {"value": 1, "timestamp": 1, "_id": 0}
            ).sort("timestamp", -1))  # Sort by timestamp descending
            
            if results:
                measurements[measurement_type] = results
                
        return measurements