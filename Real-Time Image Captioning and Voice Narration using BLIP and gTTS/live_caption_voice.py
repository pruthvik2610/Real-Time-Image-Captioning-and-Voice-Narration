import cv2
import torch
import threading
import queue
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from playsound import playsound
import tempfile

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Shared state
latest_frame = None
caption = "Starting..."
running = True
lock = threading.Lock()
spoken_captions = queue.Queue()

# TTS thread using gTTS
def voice_thread():
    while running:
        try:
            text = spoken_captions.get(timeout=1)
            if text.strip():
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts = gTTS(text=text, lang="en")
                    tts.save(fp.name)
                    playsound(fp.name)
                    os.remove(fp.name)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS error] {e}")

# Captioning thread
def captioning_thread():
    global caption, latest_frame
    last_caption = ""
    while running:
        try:
            if latest_frame is not None:
                with lock:
                    frame_copy = latest_frame.copy()

                image = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)).resize((224, 224))
                inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

                with torch.no_grad():
                    out = model.generate(**inputs)
                    result = processor.decode(out[0], skip_special_tokens=True)

                with lock:
                    if result != last_caption:
                        caption = result
                        last_caption = result
                        spoken_captions.put(result)
        except Exception as e:
            print(f"[Captioning error] {e}")

# Start webcam
cap = cv2.VideoCapture(0)

# Start threads
caption_thread = threading.Thread(target=captioning_thread)
voice_thread = threading.Thread(target=voice_thread)
caption_thread.start()
voice_thread.start()

print("Press 'q' to quit.")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with lock:
            latest_frame = frame.copy()
            display_caption = caption[:100]

        cv2.putText(frame, display_caption, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Live Caption + Voice | Press 'q' to quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Live Caption with Voice", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    caption_thread.join()
    voice_thread.join()
