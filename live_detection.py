import cv2
from ultralytics import YOLO
from datetime import datetime
import os

# Configurações
MODEL_PATH = "runs/detect/train/weights/best.pt"
OUTPUT_DIR = "processed_videos"

# Carrega modelo
model = YOLO(MODEL_PATH)

def process_video(input_path):
    # Cria diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Nome do arquivo de saída
    output_path = os.path.join(OUTPUT_DIR, f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    
    # Captura vídeo
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # VideoWriter para salvar
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detecção
        results = model(frame, conf=0.7)  # Aumente conf para reduzir falsos positivos
        annotated_frame = results[0].plot()  # Frame com bounding boxes
        
        # Exibe ao vivo
        cv2.imshow('EPI Detection', annotated_frame)
        out.write(annotated_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

if __name__ == "__main__":
    video_path = input("Digite o caminho do vídeo: ")
    if os.path.exists(video_path):
        print(f"Processando: {video_path}")
        output = process_video(video_path)
        print(f"✅ Vídeo processado salvo em: {output}")
    else:
        print("❌ Arquivo não encontrado")