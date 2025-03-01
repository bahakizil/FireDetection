import os
import cv2
import torch
import gradio as gr
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def detect_fire_in_video(input_video_path: str, output_video_path: str) -> str:
    """
    Verilen video üzerinde YOLO modeli kullanarak yangın tespiti yapar.
    Her kareye anotasyon ekler ve çıktıyı bir video dosyasına kaydeder.

    Args:
        input_video_path (str): Giriş video dosyasının yolu.
        output_video_path (str): Anotasyonlu çıktı videosunun kaydedileceği yol.

    Returns:
        str: Anotasyonlu çıktı videosunun yolu.
    """

    # Kullanılacak cihazı belirle
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Seçilen cihaz: {device}")

    # Aynı dizindeki 'last.pt' modelini yükle
    model_path = os.path.join(os.path.dirname(__file__), 'last.pt')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    model = YOLO(model_path)
    model.to(device)

    # Giriş videosunu aç
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Video dosyası açılamadı: {input_video_path}")

    # Video özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Çıktı videosu için VideoWriter oluştur
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Nesne tespiti yap
        results = model(frame, device=device)

        # Kare üzerinde anotasyon yapmak için Annotator oluştur
        annotator = Annotator(frame)

        # Tespit edilen nesneler üzerinde iterasyon yap ve anotasyon ekle
        for result in results:
            boxes = result.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()
                c = int(box.cls[0])
                label = f'{model.names[c]} {box.conf[0]:.2f}'
                annotator.box_label(b, label, color=colors(c, True))

        # Anotasyonlu kareyi çıktı videosuna yaz
        out.write(frame)

    # Kaynakları serbest bırak
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path

def process_video(input_video) -> str:
    """
    Yüklenen videoyu yangın tespiti için işler.

    Args:
        input_video (str): Yüklenen video dosyasının yolu.

    Returns:
        str: Anotasyonlu çıktı videosunun yolu.
    """
    output_video_path = "annotated_output.mp4"
    return detect_fire_in_video(input_video, output_video_path)

# Gradio arayüzünü tanımla
with gr.Blocks() as demo:
    gr.Markdown("## Yangın Tespiti Video İşleme")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Video Yükle")
            process_button = gr.Button("Videoyu İşle")
        with gr.Column():
            video_output = gr.Video(label="Anotasyonlu Video")

    process_button.click(
        fn=process_video,
        inputs=video_input,
        outputs=video_output
    )

if __name__ == "__main__":
    demo.launch()
