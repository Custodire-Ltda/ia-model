from ultralytics import YOLO
import cv2

# 1. Carrega os dois modelos
model_epi = YOLO("runs/detect/epi_treinamento/weights/best.pt")
model_train = YOLO("runs/detect/train/weights/best.pt")

# 2. Define o vídeo de teste (substitua pelo seu caminho)
video_path = "videos/istockphoto-1354886140-640_adpp_is.mp4"  # Vídeo de 10-30 segundos é ideal

# 3. Processa com ambos os modelos
def process_video(model, video_path, window_name="Resultado"):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame)[0]  # Detecção
        annotated_frame = results.plot()  # Frame com bounding boxes
        
        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 4. Roda os testes (um de cada vez)
print("=== TESTANDO MODELO EPI_TREINAMENTO ===")
process_video(model_epi, video_path, "EPI Treinamento")

print("=== TESTANDO MODELO TRAIN ===")
process_video(model_train, video_path, "Train")

# 5. Comparação numérica (opcional)
print("\nMétricas comparativas:")
print(f"EPI - Velocidade: {model_epi.predict(video_path)[0].speed['inference']}ms/frame")
print(f"Train - Velocidade: {model_train.predict(video_path)[0].speed['inference']}ms/frame")