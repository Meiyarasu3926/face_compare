# from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from typing import Optional, List, Dict
# import os
# import json
# from deepface import DeepFace
# from sklearn.metrics.pairwise import cosine_similarity
# from PIL import Image
# import numpy as np
# import cv2
# import logging
# from datetime import datetime
# import mysql.connector
# from mysql.connector import pooling
# import pickle
# from io import BytesIO
# from dotenv import load_dotenv
# import ssl
# import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # Load environment variables
# load_dotenv()

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('face_recognition.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# class TiDBConfig:
#     """TiDB specific configuration"""
#     dbconfig = {
#         "pool_name": "tidb_pool",
#         "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
#         "host": os.getenv("TIDB_HOST"),
#         "port": int(os.getenv("TIDB_PORT", "4000")),
#         "user": os.getenv("TIDB_USER"),
#         "password": os.getenv("TIDB_PASSWORD"),
#         "database": os.getenv("TIDB_DATABASE"),
#         "charset": 'utf8mb4',
#         "connect_timeout": 30000,  # 30 seconds
#         "autocommit": True,
#     }

#     # Add SSL config if CA certificate is provided
#     # if os.getenv("TIDB_SSL_CA"):
#     #     ssl_config = {
#     #         "ssl": {
#     #             "ca": os.getenv("TIDB_SSL_CA"),
#     #             "verify_cert": True
#     #         }
#     #     }
#     #     dbconfig.update(ssl_config)

#     # # Clean None values
#     # dbconfig = {k: v for k, v in dbconfig.items() if v is not None}


# class FaceRecognitionConfig:
#     TEMP_FOLDER = "/tmp"  # Use /tmp for cloud platforms
#     MODEL_NAME = "Facenet512"
#     SIMILARITY_THRESHOLD = 0.6
#     MAX_CONCURRENT_PROCESSES = 4
#     MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
#     ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# class TiDBManager:
#     def __init__(self):
#         self.setup_retries = 3
#         self.setup_database()

#     def setup_database(self):
#         """Setup TiDB connection with retry mechanism"""
#         for attempt in range(self.setup_retries):
#             try:
#                 self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(**TiDBConfig.dbconfig)
#                 self.init_database()
#                 logger.info("TiDB connection established successfully")
#                 break
#             except mysql.connector.Error as err:
#                 logger.error(f"TiDB connection attempt {attempt + 1} failed: {err}")
#                 if attempt == self.setup_retries - 1:
#                     raise
#                 time.sleep(5)

#     def init_database(self):
#         """Initialize TiDB tables with appropriate settings"""
#         connection = self.connection_pool.get_connection()
#         cursor = connection.cursor()
#         try:
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS faces (
#                     id BIGINT AUTO_RANDOM PRIMARY KEY,
#                     name VARCHAR(255) UNIQUE NOT NULL,
#                     embedding MEDIUMBLOB NOT NULL,
#                     model_name VARCHAR(50) NOT NULL,
#                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
#                     KEY idx_model_name (model_name),
#                     KEY idx_created_at (created_at)
#                 ) ENGINE = InnoDB
#             """)
#             connection.commit()
#         finally:
#             cursor.close()
#             connection.close()

#     def get_connection(self):
#         """Get TiDB connection with retry mechanism"""
#         for attempt in range(3):
#             try:
#                 conn = self.connection_pool.get_connection()
#                 # Set session variables for optimal TiDB performance
#                 cursor = conn.cursor()
#                 cursor.execute("SET tidb_disable_txn_auto_retry = 0")
#                 cursor.execute("SET tidb_retry_limit = 10")
#                 cursor.close()
#                 return conn
#             except mysql.connector.Error as err:
#                 logger.error(f"Failed to get connection, attempt {attempt + 1}: {err}")
#                 if attempt == 2:
#                     raise
#                 time.sleep(1)

#     def register_face(self, name: str, embedding: np.ndarray, model_name: str) -> bool:
#         connection = self.get_connection()
#         cursor = connection.cursor()
#         try:
#             embedding_bytes = pickle.dumps(embedding, protocol=4)
#             cursor.execute("""
#                 INSERT INTO faces (name, embedding, model_name)
#                 VALUES (%s, %s, %s)
#                 ON DUPLICATE KEY UPDATE 
#                 embedding = VALUES(embedding),
#                 model_name = VALUES(model_name),
#                 updated_at = CURRENT_TIMESTAMP
#             """, (name, embedding_bytes, model_name))
#             connection.commit()
#             return True
#         except Exception as e:
#             logger.error(f"Error registering face: {e}")
#             return False
#         finally:
#             cursor.close()
#             connection.close()

#     def get_all_faces(self) -> List[Dict]:
#         connection = self.get_connection()
#         cursor = connection.cursor(dictionary=True)
#         try:
#             cursor.execute("""
#                 SELECT name, created_at, updated_at 
#                 FROM faces
#                 ORDER BY created_at DESC
#             """)
#             results = cursor.fetchall()
#             for result in results:
#                 result['created_at'] = result['created_at'].isoformat()
#                 result['updated_at'] = result['updated_at'].isoformat()
#             return results
#         finally:
#             cursor.close()
#             connection.close()

#     def get_all_embeddings(self, model_name: str) -> List[Dict]:
#         connection = self.get_connection()
#         cursor = connection.cursor(dictionary=True)
#         try:
#             cursor.execute("""
#                 SELECT name, embedding, model_name
#                 FROM faces
#                 WHERE model_name = %s
#             """, (model_name,))
#             results = cursor.fetchall()
#             for result in results:
#                 result['embedding'] = pickle.loads(result['embedding'])
#             return results
#         finally:
#             cursor.close()
#             connection.close()

#     def delete_face(self, name: str) -> bool:
#         connection = self.get_connection()
#         cursor = connection.cursor()
#         try:
#             cursor.execute("DELETE FROM faces WHERE name = %s", (name,))
#             connection.commit()
#             return cursor.rowcount > 0
#         finally:
#             cursor.close()
#             connection.close()

# class FaceRecognitionService:
#     def __init__(self):
#         self.config = FaceRecognitionConfig()
#         self.db = TiDBManager()
#         os.makedirs(self.config.TEMP_FOLDER, exist_ok=True)

#     def validate_image(self, file: UploadFile) -> bool:
#         try:
#             file.file.seek(0, 2)
#             size = file.file.tell()
#             file.file.seek(0)
            
#             if size > self.config.MAX_FILE_SIZE:
#                 raise HTTPException(status_code=400, detail="File size too large")
            
#             ext = os.path.splitext(file.filename)[1].lower()
#             if ext not in self.config.ALLOWED_EXTENSIONS:
#                 raise HTTPException(status_code=400, detail="Invalid file type")
            
#             return True
#         except Exception as e:
#             logger.error(f"Error validating image: {e}")
#             raise HTTPException(status_code=400, detail=str(e))

#     def calculate_embedding(self, image_path: str) -> tuple:
#         try:
#             embedding = DeepFace.represent(
#                 img_path=image_path,
#                 model_name=self.config.MODEL_NAME,
#                 enforce_detection=True,
#                 detector_backend='retinaface'
#             )
#             if embedding and len(embedding) > 0:
#                 return np.array(embedding[0]["embedding"]), None
#             return None, "No face detected in the image"
#         except Exception as e:
#             logger.error(f"Error calculating embedding: {e}")
#             return None, str(e)

#     def save_upload_file(self, upload_file: UploadFile) -> str:
#         try:
#             self.validate_image(upload_file)
#             suffix = os.path.splitext(upload_file.filename)[1]
#             temp_file = os.path.join(self.config.TEMP_FOLDER, f"temp_{os.urandom(8).hex()}{suffix}")
            
#             image = Image.open(upload_file.file)
#             image = image.convert('RGB')
#             image.thumbnail((800, 800))
#             image.save(temp_file, quality=85, optimize=True)
            
#             return temp_file
#         except Exception as e:
#             logger.error(f"Error saving upload file: {e}")
#             raise HTTPException(status_code=500, detail=str(e))

# class FaceRecognitionAPI:
#     def __init__(self):
#         self.app = FastAPI(title="Face Recognition API")
#         self.service = FaceRecognitionService()
        
#         self.app.add_middleware(
#             CORSMiddleware,
#             allow_origins=["*"],
#             allow_credentials=True,
#             allow_methods=["*"],
#             allow_headers=["*"],
#         )
        
#         self.app.mount("/static", StaticFiles(directory="static"), name="static")
#         self.register_routes()

#     def register_routes(self):
#         @self.app.get("/")
#         def read_root():
#             return FileResponse('static/index.html')

#         @self.app.post("/register/")
#         def register_face(
#             name: str = Form(...),
#             file: UploadFile = File(...)
#         ):
#             temp_path = None
#             try:
#                 temp_path = self.service.save_upload_file(file)
#                 embedding, error = self.service.calculate_embedding(temp_path)
                
#                 if error:
#                     raise HTTPException(status_code=400, detail=error)
                
#                 if embedding is None:
#                     raise HTTPException(status_code=400, detail="Failed to generate embedding")

#                 success = self.service.db.register_face(
#                     name=name,
#                     embedding=embedding,
#                     model_name=self.service.config.MODEL_NAME
#                 )

#                 if not success:
#                     raise HTTPException(status_code=500, detail="Failed to register face")

#                 return {"message": "Face registered successfully", "name": name}
                
#             except Exception as e:
#                 logger.error(f"Error in register_face: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
#             finally:
#                 if temp_path and os.path.exists(temp_path):
#                     try:
#                         os.remove(temp_path)
#                     except Exception as e:
#                         logger.error(f"Error removing temp file {temp_path}: {e}")

#         @self.app.post("/compare/")
#         def compare_face(file: UploadFile = File(...)):
#             temp_path = None
#             try:
#                 temp_path = self.service.save_upload_file(file)
#                 embedding, error = self.service.calculate_embedding(temp_path)
                
#                 if error:
#                     raise HTTPException(status_code=400, detail=error)

#                 if embedding is None:
#                     raise HTTPException(status_code=400, detail="Failed to generate embedding")

#                 registered_faces = self.service.db.get_all_embeddings(self.service.config.MODEL_NAME)
                
#                 results = []
#                 for face in registered_faces:
#                     try:
#                         similarity = float(cosine_similarity([embedding], [face['embedding']])[0][0])
#                         if similarity >= self.service.config.SIMILARITY_THRESHOLD:
#                             results.append({
#                                 "name": face["name"],
#                                 "similarity": similarity,
#                                 "confidence": f"{similarity * 100:.2f}%"
#                             })
#                     except Exception as e:
#                         logger.error(f"Error processing face {face['name']}: {e}")
#                         continue

#                 return {
#                     "matches": sorted(results, key=lambda x: x["similarity"], reverse=True),
#                     "total_matches": len(results)
#                 }
                
#             except Exception as e:
#                 logger.error(f"Error in compare_face: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))
#             finally:
#                 if temp_path and os.path.exists(temp_path):
#                     try:
#                         os.remove(temp_path)
#                     except Exception as e:
#                         logger.error(f"Error removing temp file {temp_path}: {e}")

#         @self.app.get("/faces/")
#         def list_faces():
#             try:
#                 faces = self.service.db.get_all_faces()
#                 return {"faces": faces, "total": len(faces)}
#             except Exception as e:
#                 logger.error(f"Error in list_faces: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))

#         @self.app.delete("/faces/{name}")
#         def remove_face(name: str):
#             try:
#                 if self.service.db.delete_face(name):
#                     return {"message": f"Face registered as '{name}' has been removed"}
#                 raise HTTPException(status_code=404, detail=f"No face found registered as '{name}'")
#             except Exception as e:
#                 logger.error(f"Error in remove_face: {e}")
#                 raise HTTPException(status_code=500, detail=str(e))

# app = FaceRecognitionAPI().app

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 4000))
#     uvicorn.run(app, host="0.0.0.0", port=port)






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
                    detector_backend='retinaface'
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



























