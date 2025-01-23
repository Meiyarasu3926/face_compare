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
    SIMILARITY_THRESHOLD = 0.6
    MAX_CONCURRENT_PROCESSES = 4
    CLEANUP_INTERVAL = 300  # 5 minutes
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

class FaceRecognitionService:
    def __init__(self):
        self.config = FaceRecognitionConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_CONCURRENT_PROCESSES)
        self.embedding_cache = {}
        self.cache_lock = threading.Lock()
        
        # Create necessary folders
        for folder in [self.config.REGISTER_FOLDER, self.config.TEMP_FOLDER]:
            os.makedirs(folder, exist_ok=True)

    async def cleanup_temp_files(self):
        """Clean up temporary files asynchronously"""
        while True:
            try:
                current_time = datetime.now().timestamp()
                for filename in os.listdir(self.config.TEMP_FOLDER):
                    file_path = os.path.join(self.config.TEMP_FOLDER, filename)
                    if os.path.isfile(file_path):
                        file_creation_time = os.path.getctime(file_path)
                        if current_time - file_creation_time > 3600:  # Remove files older than 1 hour
                            os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
            await asyncio.sleep(self.config.CLEANUP_INTERVAL)

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

    async def calculate_embedding(self, image_path: str) -> tuple:
        """Calculate face embedding using thread pool"""
        try:
            def _calculate():
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.config.MODEL_NAME,
                    enforce_detection=True,
                    detector_backend='retinaface', 
                )
                if embedding and len(embedding) > 0:
                    return np.array(embedding[0]["embedding"]), None
                return None, "No face detected in the image"

            # Execute in thread pool
            embedding, error = await asyncio.get_event_loop().run_in_executor(
                self.executor, _calculate
            )
            return embedding, error
        except Exception as e:
            logger.error(f"Error calculating embedding: {e}")
            return None, str(e)

    async def save_upload_file(self, upload_file: UploadFile) -> str:
        """Save uploaded file with validation"""
        try:
            self.validate_image(upload_file)
            suffix = os.path.splitext(upload_file.filename)[1]
            temp_file = os.path.join(self.config.TEMP_FOLDER, f"temp_{os.urandom(8).hex()}{suffix}")
            
            # Optimize image before saving
            image = Image.open(upload_file.file)
            image = image.convert('RGB')
            image.thumbnail((800, 800))  # Resize if too large
            image.save(temp_file, quality=85, optimize=True)
            
            return temp_file
        except Exception as e:
            logger.error(f"Error saving upload file: {e}")
            raise HTTPException(status_code=500, detail=str(e))

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
        @self.app.on_event("startup")
        async def startup_event():
            asyncio.create_task(self.service.cleanup_temp_files())

        @self.app.get("/")
        async def read_root():
            return FileResponse('static/index.html')

        @self.app.post("/register/")
        async def register_face(
            name: str = Form(...),
            file: UploadFile = File(...)
        ):
            temp_path = None
            try:
                temp_path = await self.service.save_upload_file(file)
                
                embedding, error = await self.service.calculate_embedding(temp_path)
                if error:
                    raise HTTPException(status_code=400, detail=error)
                
                if embedding is None:
                    raise HTTPException(status_code=400, detail="Failed to generate embedding")

                # Save embedding with metadata
                data = {
                    "name": name,
                    "embedding": embedding.tolist(),
                    "model": self.service.config.MODEL_NAME,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                file_path = os.path.join(self.service.config.REGISTER_FOLDER, f"{name}.json")
                with open(file_path, "w") as f:
                    json.dump(data, f)
                
                return {"message": "Face registered successfully", "name": name}
                
            except Exception as e:
                logger.error(f"Error in register_face: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"Error removing temp file {temp_path}: {e}")

        @self.app.post("/compare/")
        async def compare_face(file: UploadFile = File(...)):
            """Compare face with optimized matching"""
            temp_path = None
            try:
                temp_path = await self.service.save_upload_file(file)
                
                embedding, error = await self.service.calculate_embedding(temp_path)
                if error:
                    raise HTTPException(status_code=400, detail=error)

                if embedding is None:
                    raise HTTPException(status_code=400, detail="Failed to generate embedding")

                results = []
                for file_name in os.listdir(self.service.config.REGISTER_FOLDER):
                    try:
                        if not file_name.endswith('.json'):
                            continue
                            
                        file_path = os.path.join(self.service.config.REGISTER_FOLDER, file_name)
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if data.get("model") != self.service.config.MODEL_NAME:
                                continue
                                
                            registered_embedding = np.array(data["embedding"])
                            similarity = float(cosine_similarity([embedding], [registered_embedding])[0][0])
                            
                            if similarity >= self.service.config.SIMILARITY_THRESHOLD:
                                results.append({
                                    "name": data["name"],
                                    "similarity": similarity,
                                    "confidence": f"{similarity * 100:.2f}%"
                                })
                    except Exception as e:
                        logger.error(f"Error processing {file_name}: {e}")
                        continue

                return {
                    "matches": sorted(results, key=lambda x: x["similarity"], reverse=True),
                    "total_matches": len(results)
                }
                
            except Exception as e:
                logger.error(f"Error in compare_face: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"Error removing temp file {temp_path}: {e}")

        @self.app.get("/faces/")
        async def list_faces():
            """List registered faces with metadata"""
            try:
                faces = []
                for file_name in os.listdir(self.service.config.REGISTER_FOLDER):
                    if file_name.endswith('.json'):
                        with open(os.path.join(self.service.config.REGISTER_FOLDER, file_name), 'r') as f:
                            data = json.load(f)
                            faces.append({
                                "name": data["name"],
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at")
                            })
                return {"faces": faces, "total": len(faces)}
            except Exception as e:
                logger.error(f"Error in list_faces: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/faces/{name}")
        async def remove_face(name: str):
            """Remove a registered face"""
            try:
                file_path = os.path.join(self.service.config.REGISTER_FOLDER, f"{name}.json")
                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail=f"No face found registered as '{name}'")
                    
                os.remove(file_path)
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
