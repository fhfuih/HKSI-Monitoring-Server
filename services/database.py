from pymongo import MongoClient, ASCENDING
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, List, Dict, Any
import os
import logging

logger = logging.getLogger("HKSI WebRTC")

class DatabaseService:
    def __init__(self, connection_string: Optional[str] = None):
        if connection_string is None:
            connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        self.client = MongoClient(connection_string)
        self.db = self.client.athlete_monitoring
        
        # Create indexes
        self._setup_indexes()

    def _setup_indexes(self):
        """Setup necessary indexes for performance"""
        try:
            self.db.measurements.create_index([
                ("person_id", ASCENDING),
                ("type", ASCENDING),
                ("timestamp", ASCENDING),
                ("is_final", ASCENDING)
            ])
            self.db.face_embeddings.create_index("person_id", unique=True)
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")

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

    def store_measurement(
        self, 
        person_id: str, 
        measurement_type: str, 
        value: Any, 
        timestamp: int,
        is_final: bool = False
    ):
        """Store measurement with validation and error handling"""
        try:
            # Basic validation
            if value is None or not self._validate_measurement(measurement_type, value):
                logger.warning(f"Invalid measurement value for {measurement_type}: {value}")
                return

            # Store with additional metadata
            self.db.measurements.insert_one({
                "person_id": person_id,
                "type": measurement_type,
                "value": value,
                "timestamp": timestamp,
                "is_final": is_final,
                "created_at": datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Failed to store measurement: {str(e)}")

    def _validate_measurement(self, measurement_type: str, value: Any) -> bool:
        """Validate measurement values based on type"""
        try:
            if measurement_type == "heart_rate":
                return isinstance(value, (int, float)) and 30 <= value <= 220
            elif measurement_type == "heart_rate_variability":
                return isinstance(value, (int, float)) and 0 <= value <= 200
            elif measurement_type == "fatigue":
                return isinstance(value, (int, float)) and 0 <= value <= 1
            elif measurement_type in ["darkCircleLeft", "darkCircleRight"]:
                return isinstance(value, bool) or (isinstance(value, (int, float)) and 0 <= value <= 1)
            elif measurement_type == "pimpleCount":
                return isinstance(value, int) and value >= 0
            # Add validation for wellness metrics
            elif measurement_type in ["weight", "body_fat"]:
                return isinstance(value, (int, float)) and value > 0
            elif measurement_type in ["muscle_soreness", "stress", "mood_state", "energy_levels", "sleep_quality"]:
                return isinstance(value, int) and 1 <= value <= 5
            return True
        except:
            return False

    def store_wellness_data_int(self, person_id: str, data: Dict[str, int], timestamp: int):
        """Store wellness data received from frontend"""
        try:
            # Map frontend keys to database keys
            wellness_mapping = {
                # "Weight": "weight",
                # "Body Fat": "body_fat",
                "Muscle Soreness": "muscle_soreness",
                "Stress": "stress", 
                "Mood State": "mood_state",
                "Energy Levels": "energy_levels",
                "Sleep Quality": "sleep_quality"
            }
            
            # Store each wellness metric
            for frontend_key, db_key in wellness_mapping.items():
                if frontend_key in data:
                    value = data[frontend_key]
                    if self._validate_measurement(db_key, value):
                        self.store_measurement(
                            person_id=person_id,
                            measurement_type=db_key,
                            value=value,
                            timestamp=timestamp,
                            is_final=True  # Wellness data is always considered final
                        )
                    else:
                        logger.warning(f"Invalid wellness value for {db_key}: {value}")
        except Exception as e:
            logger.error(f"Failed to store wellness data int: {str(e)}")


    def store_wellness_data_float(self, person_id: str, data: Dict[str, float], timestamp: int):
        """Store wellness data received from frontend"""
        try:
            # Map frontend keys to database keys
            wellness_mapping = {
                "Weight": "weight",
                "Body Fat": "body_fat"
            }

            # Store each wellness metric
            for frontend_key, db_key in wellness_mapping.items():
                if frontend_key in data:
                    value = data[frontend_key]
                    if self._validate_measurement(db_key, value):
                        self.store_measurement(
                            person_id=person_id,
                            measurement_type=db_key,
                            value=value,
                            timestamp=timestamp,
                            is_final=True  # Wellness data is always considered final
                        )
                    else:
                        logger.warning(f"Invalid wellness value for {db_key}: {value}")
        except Exception as e:
            logger.error(f"Failed to store wellness data float: {str(e)}")


    def get_person_measurements_summary(
        self, 
        person_id: str,
        only_final: bool = True,
        limit: int = 100
    ) -> Dict[str, List[Dict]]:
        """Get measurements summary with filtering options"""
        try:
            measurements = {}
            measurement_types = [
                'heart_rate', 'heart_rate_variability', 'fatigue', 
                'darkCircleLeft', 'darkCircleRight', 'pimpleCount',
                # Add wellness metrics
                'weight', 'body_fat', 'muscle_soreness', 'stress',
                'mood_state', 'energy_levels', 'sleep_quality'
            ]
            
            for measurement_type in measurement_types:
                query = {
                    "person_id": person_id,
                    "type": measurement_type
                }
                
                if only_final:
                    query["is_final"] = True

                results = list(self.db.measurements.find(
                    query,
                    {"value": 1, "timestamp": 1, "is_final": 1, "_id": 0}
                ).sort("timestamp", -1).limit(limit))
                
                if results:
                    measurements[measurement_type] = results
                    
            return measurements
        except Exception as e:
            logger.error(f"Failed to get measurements summary: {str(e)}")
            return {}

    def cleanup_old_measurements(self, days_to_keep: int = 400):
        """Cleanup old measurements"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            self.db.measurements.delete_many({
                "created_at": {"$lt": cutoff_date},
                "is_final": False  # Only delete intermediate measurements
            })
        except Exception as e:
            logger.error(f"Failed to cleanup old measurements: {str(e)}")