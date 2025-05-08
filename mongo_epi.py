from pymongo import MongoClient
from datetime import datetime
import uuid
import os
from ultralytics import YOLO
import numpy as np
import cv2

# Conexão com MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["epi_detection"]
detections = db["detections"]

def post_process(detections, img_width=640, img_height=640):
    """Post-processing compatível com Ultralytics 8.3.96"""
    if not detections["objects"]:
        return detections
    
    # Converter para arrays numpy
    boxes = np.array([obj["bbox"] for obj in detections["objects"]])
    scores = np.array([obj["confidence"] for obj in detections["objects"]])
    
    # Implementação manual do NMS
    def nms(boxes, scores, iou_threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    # Aplicar NMS manual
    keep = nms(boxes, scores, iou_threshold=0.45)
    
    # Processar resultados
    filtered = []
    for i in keep:
        obj = detections["objects"][i]
        obj["class"] = "capacete" if obj["class"].lower() in ["capacete", "helmet"] else "sem_capacete"
        obj["bbox"] = [
            boxes[i][0]/img_width,
            boxes[i][1]/img_height,
            boxes[i][2]/img_width,
            boxes[i][3]/img_height
        ]
        filtered.append(obj)
    
    return {"objects": filtered}

def save_detection(video_path, frame_data, frame_number=None):
    """Versão robusta com fallback para arquivo"""
    try:
        if not frame_data["objects"]:
            return False
            
        doc = {
            "_id": str(uuid.uuid4()),
            "video": os.path.basename(video_path),
            "timestamp": datetime.now(),
            "frame_number": frame_number,
            "detections": frame_data["objects"],
            "confidence_avg": sum(obj["confidence"] for obj in frame_data["objects"])/len(frame_data["objects"]),
            "needs_review": any(obj["confidence"] < 0.7 or obj["class"] == "sem_capacete" for obj in frame_data["objects"])
        }
        
        # Tenta inserir no MongoDB
        try:
            result = detections.insert_one(doc)
            print(f"✅ Frame {frame_number} - Salvo (ID: {result.inserted_id})")
            return True
        except Exception as e:
            print(f"⚠️ Falha no MongoDB: {e}")
            # Fallback para arquivo
            with open("detections_backup.json", "a") as f:
                f.write(f"{doc}\n")
            return False
            
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        return False

def process_video(video_path, show_output=False):
    """Processamento otimizado para versão 8.x"""
    try:
        model_path = "runs/detect/epi_treinamento/weights/best.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
            
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 5 != 0:  # Processa 1 a cada 5 frames
                continue
                
            # Detecção
            results = model(frame, conf=0.6, verbose=False)
            
            # Extrai resultados
            frame_data = {"objects": []}
            for box in results[0].boxes:
                frame_data["objects"].append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })
            
            # Pós-processamento
            processed = post_process(frame_data, frame.shape[1], frame.shape[0])
            if processed["objects"]:
                save_detection(video_path, processed, frame_count)
                
                # Visualização
                if show_output:
                    annotated = results[0].plot()
                    cv2.imshow('EPI Detection', annotated)
                    if cv2.waitKey(1) == ord('q'):
                        break
                        
        cap.release()
        if show_output:
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")

def main():
    video_test = "video/r5.mp4" 
    
    if os.path.exists(video_test):
        print(f"🔍 Processando: {video_test}")
        process_video(video_test, show_output=True)
        
        # Resumo
        total = detections.count_documents({"video": os.path.basename(video_test)})
        needs_review = detections.count_documents({
            "video": os.path.basename(video_test),
            "needs_review": True
        })
        print(f"\n📊 Resumo:")
        print(f"- Frames processados: {total}")
        print(f"- Necessitando revisão: {needs_review}")
    else:
        print(f"❌ Vídeo não encontrado: {video_test}")

if __name__ == "__main__":
    main()