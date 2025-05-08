from ultralytics import YOLO
import os

# Carrega o modelo treinado
model = YOLO("runs/detect/train/weights/best.pt")

# Pasta contendo os vídeos
video_dir = "/Users/Julliano Rodrigues/Desktop/IA/videos"

# Processa cada vídeo da pasta
for video_file in os.listdir(video_dir):
    if video_file.endswith((".mp4", ".avi", ".mov")):
        video_path = os.path.join(video_dir, video_file)
        # Teste no primeiro vídeo encontrado
        results = model.predict(source=video_path, save=True, show=True)
        print(f"Processado: {video_file}")