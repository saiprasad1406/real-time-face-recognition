import cv2
import face_recognition
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import os
import dlib

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_metadata = []
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the data directory"""
        # Read user data from Excel
        df = pd.read_excel('data/face_data.xlsx')
        
        for _, row in df.iterrows():
            image_path = f"data/faces/{row['image_filename']}"
            if os.path.exists(image_path):
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_metadata.append({
                        'name': row['name'],
                        'age': row['age'],
                        'gender': row['gender'],
                        'phone': row['phone'],
                        'aadhar': row['aadhar']
                    })

    def process_frame(self, frame):
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if not face_locations:
                return frame, None
            
            # Get face landmarks and create face_encodings properly
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
            face_encodings = []
            
            # Create a dlib face detector and shape predictor
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
            
            # Convert face_locations to dlib rectangles
            dlib_rects = [dlib.rectangle(left, top, right, bottom) for top, right, bottom, left in face_locations]
            
            # Get face shapes and compute descriptors
            for rect in dlib_rects:
                shape = predictor(rgb_small_frame, rect)
                face_descriptor = face_rec_model.compute_face_descriptor(rgb_small_frame, shape)
                face_encodings.append(np.array(face_descriptor))
            
            # Compare with known faces
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                if len(self.known_face_encodings) > 0:
                    # Compare faces
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            # Get matching person's data
                            person_data = self.known_face_metadata[best_match_index]
                            match_percentage = (1 - face_distances[best_match_index]) * 100
                            
                            # Scale back up face locations
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4
                            
                            # Draw rectangle and label
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, person_data.get('name', 'Unknown'), 
                                      (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
                            
                            recognition_results = {
                                'match_percentage': f"{match_percentage:.1f}",
                                'name': person_data.get('name', 'Unknown'),
                                'age': person_data.get('age', 'N/A'),
                                'gender': person_data.get('gender', 'N/A'),
                                'phone': person_data.get('phone', 'N/A'),
                                'aadhar': person_data.get('aadhar', 'N/A')
                            }
                            
                            return frame, recognition_results
            
            return frame, None
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return frame, None

    def add_new_face(self, image: np.ndarray, metadata: Dict) -> bool:
        """
        Add a new face to the recognition system
        """
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) == 0:
            return False
        
        # Save face encoding and metadata
        self.known_face_encodings.append(face_encodings[0])
        self.known_face_metadata.append(metadata)
        
        # Save to Excel
        df = pd.DataFrame(self.known_face_metadata)
        df.to_excel('data/face_data.xlsx', index=False)
        
        # Save face image
        image_filename = f"{metadata['name'].lower().replace(' ', '_')}_{len(self.known_face_encodings)}.jpg"
        cv2.imwrite(f"data/faces/{image_filename}", image)
        
        return True 