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
from datetime import datetime
import mysql.connector
from mysql.connector import pooling
import pickle
from io import BytesIO
from dotenv import load_dotenv
import ssl
import time
import tensorflow as tf

# Load environment variables
load_dotenv()

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

class TiDBConfig:
    """TiDB specific configuration"""
    dbconfig = {
        "pool_name": "tidb_pool",
        "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),
        "host": os.getenv("TIDB_HOST"),
        "port": int(os.getenv("TIDB_PORT", "4000")),
        "user": os.getenv("TIDB_USER"),
        "password": os.getenv("TIDB_PASSWORD"),
        "database": os.getenv("TIDB_DATABASE"),
        "charset": 'utf8mb4',
        "connect_timeout": 30000,  # 30 seconds
        "autocommit": True,
    }

    @classmethod
    def get_ssl_config(cls):
        """Get SSL configuration if CA certificate is provided"""
        if os.getenv("TIDB_SSL_CA"):
            return {
                "ssl": {
                    "ca": os.getenv("TIDB_SSL_CA"),
                    "verify_cert": True
                }
            }
        return {}

class FaceRecognitionConfig:
    TEMP_FOLDER = "/tmp"  # Use /tmp for cloud platforms
    MODEL_NAME = "Facenet512"
    SIMILARITY_THRESHOLD = 0.6
    MAX_CONCURRENT_PROCESSES = 4
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
    
    # GPU Configuration
    USE_GPU = False  # Will be set dynamically based on availability
    GPU_MEMORY_LIMIT = 0.7  # Use 70% of available GPU memory
    
    @classmethod
    def setup_gpu(cls):
        """Configure GPU settings and fallback to CPU if necessary"""
        try:
            # Check if GPU is available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Configure GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    # Set memory limit
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=int(cls.GPU_MEMORY_LIMIT * 1024))])
                cls.USE_GPU = True
                logger.info("GPU configured successfully")
            else:
                cls.USE_GPU = False
                logger.info("No GPU available, using CPU")
                
        except Exception as e:
            cls.USE_GPU = False
            logger.warning(f"Error configuring GPU, falling back to CPU: {e}")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class TiDBManager:
    def __init__(self):
        self.setup_retries = 3
        self.setup_database()

    def setup_database(self):
        """Setup TiDB connection with retry mechanism"""
        config = {**TiDBConfig.dbconfig, **TiDBConfig.get_ssl_config()}
        for attempt in range(self.setup_retries):
            try:
                self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(**config)
                self.init_database()
                logger.info("TiDB connection established successfully")
                break
            except mysql.connector.Error as err:
                logger.error(f"TiDB connection attempt {attempt + 1} failed: {err}")
                if attempt == self.setup_retries - 1:
                    raise
                time.sleep(5)

    def init_database(self):
        """Initialize TiDB tables with appropriate settings"""
        connection = self.connection_pool.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id BIGINT AUTO_RANDOM PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    embedding MEDIUMBLOB NOT NULL,
                    model_name VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    KEY idx_model_name (model_name),
                    KEY idx_created_at (created_at)
                ) ENGINE = InnoDB
            """)
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    def get_connection(self):
        """Get TiDB connection with retry mechanism"""
        for attempt in range(3):
            try:
                conn = self.connection_pool.get_connection()
                cursor = conn.cursor()
                cursor.execute("SET tidb_disable_txn_auto_retry = 0")
                cursor.execute("SET tidb_retry_limit = 10")
                cursor.close()
                return conn
            except mysql.connector.Error as err:
                logger.error(f"Failed to get connection, attempt {attempt + 1}: {err}")
                if attempt == 2:
                    raise
                time.sleep(1)

    def register_face(self, name: str, embedding: np.ndarray, model_name: str) -> bool:
        connection = self.get_connection()
        cursor = connection.cursor()
        try:
            embedding_bytes = pickle.dumps(embedding, protocol=4)
            cursor.execute("""
                INSERT INTO faces (name, embedding, model_name)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                embedding = VALUES(embedding),
                model_name = VALUES(model_name),
                updated_at = CURRENT_TIMESTAMP
            """, (name, embedding_bytes, model_name))
            connection.commit()
            return True
        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return False
        finally:
            cursor.close()
            connection.close()

    def get_all_faces(self) -> List[Dict]:
        connection = self.get_connection()
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT name, created_at, updated_at 
                FROM faces
                ORDER BY created_at DESC
            """)
            results = cursor.fetchall()
            for result in results:
                result['created_at'] = result['created_at'].isoformat()
                result['updated_at'] = result['updated_at'].isoformat()
            return results
        finally:
            cursor.close()
            connection.close()

    def get_all_embeddings(self, model_name: str) -> List[Dict]:
        connection = self.get_connection()
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT name, embedding, model_name
                FROM faces
                WHERE model_name = %s
            """, (model_name,))
            results = cursor.fetchall()
            for result in results:
                result['embedding'] = pickle.loads(result['embedding'])
            return results
        finally:
            cursor.close()
            connection.close()

    def delete_face(self, name: str) -> bool:
        connection = self.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("DELETE FROM faces WHERE name = %s", (name,))
            connection.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()
            connection.close()

class FaceRecognitionService:
    def __init__(self):
        self.config = FaceRecognitionConfig()
        self.db = TiDBManager()
        os.makedirs(self.config.TEMP_FOLDER, exist_ok=True)
        
        # Configure GPU settings
        FaceRecognitionConfig.setup_gpu()
        
        # Initialize DeepFace with appropriate backend
        self.initialize_deepface()

    def initialize_deepface(self):
        """Initialize DeepFace with appropriate backend settings"""
        try:
            # Set detector backend based on GPU availability
            self.detector_backend = 'retinaface'
            if not self.config.USE_GPU:
                # Use a less GPU-intensive backend for CPU
                self.detector_backend = 'opencv'
            
            # Warm up the model
            test_img = np.zeros((112, 112, 3), dtype=np.uint8)
            test_path = os.path.join(self.config.TEMP_FOLDER, 'test.jpg')
            cv2.imwrite(test_path, test_img)
            
            try:
                DeepFace.represent(
                    img_path=test_path,
                    model_name=self.config.MODEL_NAME,
                    enforce_detection=False,
                    detector_backend=self.detector_backend
                )
            finally:
                if os.path.exists(test_path):
                    os.remove(test_path)
                    
            logger.info(f"DeepFace initialized successfully with {self.detector_backend} backend")
        except Exception as e:
            logger.error(f"Error initializing DeepFace: {e}")
            raise

    def validate_image(self, file: UploadFile) -> bool:
        """Validate uploaded image file"""
        try:
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
            
            if size > self.config.MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File size too large")
            
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in self.config.ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail="Invalid file type")
            
            return True
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    def calculate_embedding(self, image_path: str) -> tuple:
        """Calculate face embedding from image"""
        try:
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name=self.config.MODEL_NAME,
                enforce_detection=True,
                detector_backend=self.detector_backend
            )
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]["embedding"]), None
            return None, "No face detected in the image"
        except Exception as e:
            logger.error(f"Error calculating embedding: {e}")
            return None, str(e)

    def save_upload_file(self, upload_file: UploadFile) -> str:
        """Save and optimize uploaded file"""
        try:
            self.validate_image(upload_file)
            suffix = os.path.splitext(upload_file.filename)[1]
            temp_file = os.path.join(self.config.TEMP_FOLDER, f"temp_{os.urandom(8).hex()}{suffix}")
            
            image = Image.open(upload_file.file)
            image = image.convert('RGB')
            image.thumbnail((800, 800))
            image.save(temp_file, quality=85, optimize=True)
            
            return temp_file
        except Exception as e:
            logger.error(f"Error saving upload file: {e}")
            raise HTTPException(status_code=500, detail=str(e))

class FaceRecognitionAPI:
    def __init__(self):
        self.app = FastAPI(title="Face Recognition API")
        self.service = FaceRecognitionService()
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.register_routes()

    def register_routes(self):
        @self.app.get("/")
        def read_root():
            return FileResponse('static/index.html')

        @self.app.post("/register/")
        async def register_face(
            name: str = Form(...),
            file: UploadFile = File(...)
        ):
            temp_path = None
            try:
                temp_path = self.service.save_upload_file(file)
                embedding, error = self.service.calculate_embedding(temp_path)
                
                if error:
                    raise HTTPException(status_code=400, detail=error)
                
                if embedding is None:
                    raise HTTPException(status_code=400, detail="Failed to generate embedding")

                success = self.service.db.register_face(
                    name=name,
                    embedding=embedding,
                    model_name=self.service.config.MODEL_NAME
                )

                if not success:
                    raise HTTPException(status_code=500, detail="Failed to register face")

                return {"message": "Face registered successfully", "name": name}
                
            except Exception as e:
                logger.error(f"Error in register_face: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"Error removing temp file {temp_path}: {e}")

        @self.app.post("/compare/")
        async def compare_face(file: UploadFile = File(...)):
            temp_path = None
            try:
                temp_path = self.service.save_upload_file(file)
                embedding, error = self.service.calculate_embedding(temp_path)
                
                if error:
                    raise HTTPException(status_code=400, detail=error)

                if embedding is None:
                    raise HTTPException(status_code=400, detail="Failed to generate embedding")

                registered_faces = self.service.db.get_all_embeddings(self.service.config.MODEL_NAME)
                
                results = []
                for face in registered_faces:
                    try:
                        similarity = float(cosine_similarity([embedding], [face['embedding']])[0][0])
                        if similarity >= self.service.config.SIMILARITY_THRESHOLD:
                            results.append({
                                "name": face["name"],
                                "similarity": similarity,
                                "confidence": f"{similarity * 100:.2f}%"
                            })
                    except Exception as e:
                        logger.error(f"Error processing face {face['name']}: {e}")
                        continue

                return {
                    "matches": sorted(results, key=lambda x: x["similarity"], reverse=True),
                    "total_matches": len(results)
                }
                
            except Exception as e:
                logger.error(f"Error in compare_face: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        logger.error(f"Error removing temp file {temp_path}: {e}")

        @self.app.get("/faces/")
        async def list_faces():
            try:
                faces = self.service.db.get_all_faces()
                return {"faces": faces, "total": len(faces)}
            except Exception as e:
                logger.error(f"Error in list_faces: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/faces/{name}")
        async def remove_face(name: str):
            try:
                if self.service.db.delete_face(name):
                    return {"message": f"Face registered as '{name}' has been removed"}
                raise HTTPException(status_code=404, detail=f"No face found registered as '{name}'")
            except Exception as e:
                logger.error(f"Error in remove_face: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            try:
                # Check database connection
                self.service.db.get_connection().close()
                
                # Check GPU status
                gpu_status = "GPU enabled" if self.service.config.USE_GPU else "CPU mode"
                
                return {
                    "status": "healthy",
                    "database": "connected",
                    "processing": gpu_status,
                    "backend": self.service.detector_backend,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Service unhealthy: {str(e)}"
                )

def create_app():
    """Application factory function"""
    try:
        return FaceRecognitionAPI().app
    except Exception as e:
        logger.critical(f"Failed to create application: {e}")
        raise

app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 4000))
    workers = int(os.getenv("WORKERS", 1))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    # Configure logging for uvicorn
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Start server
    try:
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_config=log_config,
            access_log=True
        )
    except Exception as e:
        logger.critical(f"Failed to start server: {e}")
        raise
