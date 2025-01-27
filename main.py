from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict
import os
import json
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import cv2
import logging
import shutil
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from io import BytesIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceRecognitionConfig:
    REGISTER_FOLDER = "registered_faces"
    TEMP_FOLDER = "temp_files"
    MODEL_NAME = "Facenet512"
    SIMILARITY_THRESHOLD = 0.7
    MAX_CONCURRENT_PROCESSES = 4
    CLEANUP_INTERVAL = 300  # 5 minutes
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_IMAGES_PER_PERSON = 10
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

class FaceRecognitionService:
    def __init__(self):
        self.config = FaceRecognitionConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_CONCURRENT_PROCESSES)
        
        # Create necessary folders
        for folder in [self.config.REGISTER_FOLDER, self.config.TEMP_FOLDER]:
            os.makedirs(folder, exist_ok=True)

    def validate_image(self, file: UploadFile) -> bool:
        """Validate image file"""
        try:
            # Check file size
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
            
            if size > self.config.MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File size too large")
            
            # Check file extension
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in self.config.ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Invalid file type")
            
            return True
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    def save_upload_file(self, upload_file: UploadFile, name: str) -> str:
        """Save uploaded file with validation and name-based storage"""
        try:
            self.validate_image(upload_file)
            suffix = os.path.splitext(upload_file.filename)[1]
            
            # Ensure the registered folder for the person exists
            person_folder = os.path.join(self.config.REGISTER_FOLDER, name)
            os.makedirs(person_folder, exist_ok=True)
            
            # Count existing files to limit total images
            existing_files = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            if len(existing_files) >= self.config.MAX_IMAGES_PER_PERSON:
                raise HTTPException(status_code=400, detail=f"Maximum {self.config.MAX_IMAGES_PER_PERSON} images allowed per person")
            
            # Generate unique filename
            filename = f"{name}_{len(existing_files) + 1}{suffix}"
            file_path = os.path.join(person_folder, filename)
            
            # Optimize image before saving
            image = Image.open(upload_file.file)
            image = image.convert('RGB')
            image.thumbnail((800, 800))  # Resize if too large
            image.save(file_path, quality=85, optimize=True)
            
            return file_path
        except Exception as e:
            logger.error(f"Error saving upload file: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def calculate_embedding(self, image_path: str) -> tuple:
        """Calculate face embedding"""
        try:
            embedding = DeepFace.represent(
                img_path=image_path,
                enforce_detection=True,
                model_name="VGG-Face",
                detector_backend='retinaface', 
            )
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]["embedding"]), None
            return None, "No face detected in the image"
        except Exception as e:
            logger.error(f"Error calculating embedding: {e}")
            return None, str(e)

class FaceRecognitionAPI:
    def __init__(self):
        self.app = FastAPI(title="Face Recognition API")
        self.service = FaceRecognitionService()
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Register routes
        self.register_routes()

    def register_routes(self):
        
        @self.app.get("/")
        async def read_root():
            return FileResponse('static/index.html')
        

        @self.app.post("/register/")
        async def register_face(
            name: str = Form(...),
            files: List[UploadFile] = File(...)
        ):
            try:
                registered_images = []
                embeddings_file_path = os.path.join(
                    self.service.config.REGISTER_FOLDER, 
                    f"{name}_embeddings.json"
                )

                # Prepare embeddings data
                embeddings_data = {
                    "name": name,
                    "model": self.service.config.MODEL_NAME,
                    "embeddings": [],
                    "images": [],
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }

                # Process each uploaded file
                for file in files:
                    # Save file
                    saved_path = self.service.save_upload_file(file, name)
                    registered_images.append(saved_path)

                    # Calculate embedding for each image
                    embedding, error = await self.service.calculate_embedding(saved_path)
                    if error:
                        logger.warning(f"Skipping image due to embedding error: {error}")
                        continue

                    embeddings_data["embeddings"].append(embedding.tolist())
                    embeddings_data["images"].append(os.path.basename(saved_path))

                # Save embeddings data
                with open(embeddings_file_path, "w") as f:
                    json.dump(embeddings_data, f)

                return {
                    "message": f"Registered {len(registered_images)} images for {name}", 
                    "registered_images": [os.path.basename(img) for img in registered_images]
                }
                
            except Exception as e:
                logger.error(f"Error in register_face: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/compare/")
        async def compare_face(file: UploadFile = File(...)):
            """Compare face with optimized matching, returning all similarity scores regardless of threshold"""
            try:
                # Temporary save of uploaded file
                temp_suffix = os.path.splitext(file.filename)[1]
                temp_path = os.path.join(
                    self.service.config.TEMP_FOLDER, 
                    f"temp_{os.urandom(8).hex()}{temp_suffix}"
                )
                
                self.service.validate_image(file)
                image = Image.open(file.file)
                image = image.convert('RGB')
                image.thumbnail((800, 800))
                image.save(temp_path, quality=85, optimize=True)
        
                # Calculate embedding for uploaded image
                embedding, error = await self.service.calculate_embedding(temp_path)
                if error:
                    raise HTTPException(status_code=400, detail=error)
        
                results = {}
                # Scan through all people's embedding files
                for embeddings_file in os.listdir(self.service.config.REGISTER_FOLDER):
                    if not embeddings_file.endswith('_embeddings.json'):
                        continue
                    
                    file_path = os.path.join(
                        self.service.config.REGISTER_FOLDER, 
                        embeddings_file
                    )
                    
                    with open(file_path, "r") as f:
                        data = json.load(f)
        
                    # Find the most similar image for this person
                    best_match = None
                    for idx, registered_embedding in enumerate(data["embeddings"]):
                        similarity = float(
                            cosine_similarity(
                                [embedding], 
                                [np.array(registered_embedding)]
                            )[0][0]
                        )
        
                        # Keep track of the most similar image, regardless of threshold
                        if best_match is None or similarity > best_match['similarity']:
                            best_match = {
                                "name": data["name"],
                                "similarity": similarity,
                                "confidence": f"{similarity * 100:.2f}%",
                                "image": data["images"][idx],
                                "exceeds_threshold": similarity >= self.service.config.SIMILARITY_THRESHOLD
                            }
        
                    # Add best match for this person to results
                    if best_match:
                        results[data["name"]] = best_match
        
                # Clean up temp file
                os.remove(temp_path)
        
                # Convert results to list sorted by similarity
                sorted_results = sorted(
                    results.values(), 
                    key=lambda x: x["similarity"], 
                    reverse=True
                )
        
                return {
                    "matches": sorted_results,
                    "total_matches": len([r for r in sorted_results if r["exceeds_threshold"]]),
                    "threshold": self.service.config.SIMILARITY_THRESHOLD
                }
            except Exception as e:
                logger.error(f"Error in compare_face: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
        @self.app.get("/faces/")
        async def list_faces():
            """List registered faces"""
            try:
                faces = []
                for file_name in os.listdir(self.service.config.REGISTER_FOLDER):
                    if file_name.endswith('_embeddings.json'):
                        with open(os.path.join(self.service.config.REGISTER_FOLDER, file_name), 'r') as f:
                            data = json.load(f)
                            faces.append({
                                "name": data["name"],
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "image_count": len(data.get("images", []))
                            })
                return {"faces": faces, "total": len(faces)}
            except Exception as e:
                logger.error(f"Error in list_faces: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/faces/{name}")
        async def remove_face(name: str):
            """Remove a registered face"""
            try:
                embeddings_file_path = os.path.join(
                    self.service.config.REGISTER_FOLDER, 
                    f"{name}_embeddings.json"
                )
                person_folder = os.path.join(
                    self.service.config.REGISTER_FOLDER, 
                    name
                )

                if not os.path.exists(embeddings_file_path):
                    raise HTTPException(status_code=404, detail=f"No face found registered as '{name}'")
                
                # Remove embeddings file and image folder
                os.remove(embeddings_file_path)
                shutil.rmtree(person_folder, ignore_errors=True)
                
                return {"message": f"Face registered as '{name}' has been removed"}
            except Exception as e:
                logger.error(f"Error in remove_face: {e}")
                raise HTTPException(status_code=500, detail=str(e))

# Initialize and run the application
app = FaceRecognitionAPI().app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 4000))
    uvicorn.run(app, host="0.0.0.0", port=port)
