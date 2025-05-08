from ultralytics import YOLO
import os
from pathlib import Path

# Configurações absolutas
BASE_DIR = Path("C:/Users/Julliano Rodrigues/Desktop/IA")
DATA_YAML = BASE_DIR / "dataset/data.yaml"
MODEL_PATH = BASE_DIR / "runs/detect/epi_treinamento/weights/best.pt"

# Verificação crítica
if not DATA_YAML.exists():
    raise FileNotFoundError(f"Arquivo data.yaml não encontrado em: {DATA_YAML}")

# Treinamento
model = YOLO(MODEL_PATH)
results = model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=8,  # Reduzi para evitar sobrecarga na CPU
    device='cpu',
    project=str(BASE_DIR / "runs/detect"),
    name="epi_treinamento",
    exist_ok=True,
    workers=0  # Importante para Windows
)