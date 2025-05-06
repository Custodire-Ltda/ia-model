from ultralytics import YOLO
import os

# Carrega o modelo treinado
model = YOLO("runs/detect/train/weights/best.pt")

images_dir = "/Users/Julliano Rodrigues/Desktop/IA/dataset/test/images"

# Processa todas as imagens .jpg/.png
for img_file in os.listdir(images_dir):
    if img_file.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(images_dir, img_file)
        results = model.predict(source=img_path, save=True, show=True)