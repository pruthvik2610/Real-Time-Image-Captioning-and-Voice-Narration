# 🧠 Live Image Captioning with Voice Narration


This project is a real-time **image captioning system** powered by the BLIP Transformer model and enhanced with **voice narration** using Google Text-to-Speech (gTTS). It captures frames from your webcam, generates natural language descriptions of what it sees, and reads them aloud — all in real-time!


---


## 📸 Demo


> The system launches your webcam, generates a caption for the scene every few seconds, and speaks it aloud.


![Live Caption Screenshot](./screenshot.png)


---


## 🚀 Features


- 📷 Real-time webcam frame capture
- 🤖 Image captioning using [BLIP (Salesforce)](https://huggingface.co/Salesforce/blip-image-captioning-base)
- 🔊 Voice narration using `gTTS` and `playsound`
- 🧵 Multithreaded for smooth captioning and TTS
- 🧠 GPU acceleration (if available via PyTorch CUDA)
- 🎛️ Live caption display over the video feed
- ✅ Thread-safe frame handling and error handling


---


## 🧑‍💻 Requirements


- Python 3.7+
- Torch (with CUDA if available)
- OpenCV
- Transformers (HuggingFace)
- Pillow
- gTTS
- playsound


---


## 📦 Installation


```bash
# Clone the repository
git clone https://github.com/yourusername/live-caption-voice.git
cd live-caption-voice


# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate


# Install dependencies
pip install -r requirements.txt