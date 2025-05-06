from pymongo import MongoClient
from datetime import datetime
import uuid
import os
from ultralytics import YOLO

# Conexão com MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["epi_detection"]
detections = db["detections"]

def save_detection(video_path, frame_data):
    try:
        doc = {
            "_id": str(uuid.uuid4()),
            "video": os.path.basename(video_path),
            "timestamp": datetime.now(),
            "detections": frame_data["objects"],
            "confidence_avg": sum(obj["confidence"] for obj in frame_data["objects"])/len(frame_data["objects"]),
            "needs_review": any(obj["confidence"] < 0.7 for obj in frame_data["objects"])
        }
        detections.insert_one(doc)
        print(f"✅ Detecção salva (ID: {doc['_id']})")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar: {e}")
        return False

def process_video(video_path):
    try:
        # ATENÇÃO: Verifique e atualize este caminho!
        model_path = "runs/detect/train/weights/best.pt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
            
        model = YOLO(model_path)
        
        results = model.predict(
            source=video_path,
            conf=0.6,
            stream=True
        )
        
        for frame in results:
            frame_data = {
                "objects": [{
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                } for box in frame.boxes]
            }
            
            if frame_data["objects"]:
                save_detection(video_path, frame_data)
                
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")

if __name__ == "__main__":
    video_test = "videos/istockphoto-876895990-640_adpp_is.mp4"
    
    if os.path.exists(video_test):
        print(f"🔍 Processando vídeo: {video_test}")
        process_video(video_test)
    else:
        print(f"❌ Arquivo não encontrado: {video_test}")
        print("Verifique se:")
        print("1. O vídeo está na pasta 'videos/'")
        print("2. O nome do arquivo está correto")