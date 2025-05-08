from ultralytics import YOLO

# Carrega explicitamente o último checkpoint
model = YOLO("runs/detect/epi_treinamento/weights/last.pt")

# Continua o treinamento
results = model.train(
    resume=True,
    device="cpu",
    batch=4,  # Reduza para evitar travamentos
    imgsz=320  # Reduza a resolução se necessário
)