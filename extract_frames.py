import cv2
import os

# Caminho para a pasta com os vídeos
video_folder = "/Users/Julliano Rodrigues/Desktop/IA/videos"
# Caminho para salvar os frames
output_folder = "/Users/Julliano Rodrigues/Desktop/IA/dataset/images/train"

# Certifique-se de que a pasta de saída existe
os.makedirs(output_folder, exist_ok=True)

# Taxa de extração de frames (1 frame por segundo)
fps_rate = 1  # Altere para 2, 3, etc., se quiser mais frames por segundo

# Loop através de todos os vídeos na pasta
for video_name in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_name)
    if not video_path.endswith((".mp4", ".avi", ".mov")):  # Verifique se é um vídeo
        continue

    # Abra o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo {video_name}")
        continue

    # Obtenha o FPS do vídeo
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps_rate)  # Intervalo entre frames

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Salve 1 frame por segundo (ou conforme a taxa definida)
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{saved_frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames extraídos de {video_name}: {saved_frame_count}")

print("Extração de frames concluída!")