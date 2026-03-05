# 🖼️ Watermark Removal System

A deep learning-based image inpainting system that automatically detects and removes watermarks from images — producing clean, high-quality outputs without visible artifacts.

---

## ✨ Demo

| Watermarked Image | Cleaned Output |
|:-----------------:|:--------------:|
| ![Input](https://github.com/Ammar-Ali234/Watermark-Removal-System/blob/main/test2.jpg?raw=true) | ![Output](https://github.com/Ammar-Ali234/Watermark-Removal-System/blob/main/out3.png?raw=true) |

---

## 🚀 Features

- **Automatic watermark detection** using image preprocessing
- **Deep learning inpainting** to seamlessly reconstruct image regions
- **Batch processing** support for multiple images at once
- **Guided batch mode** for user-directed watermark region selection
- **Flask web app** (`app.py`) for a simple browser-based interface
- **Docker support** for easy, reproducible deployment
- Pre-trained model weights included (via Google Drive)

---

## 🗂️ Project Structure

```
Watermark-Removal-System/
│
├── model/                    # Pre-trained model checkpoints
├── utils/
│   └── istock/landscape/     # Sample/test utility images
│
├── app.py                    # Flask web application
├── main.py                   # Single image inference entry point
├── batch_test.py             # Batch processing script
├── guided_batch_test.py      # Guided (mask-assisted) batch processing
├── inpaint_model.py          # Core inpainting model definition
├── inpaint_ops.py            # Inpainting operations and utilities
├── preprocess_image.py       # Image preprocessing pipeline
├── inpaint.yml               # Conda environment configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container setup
└── README.md
```

---

## ⚙️ Installation

### Option 1 — Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/Ammar-Ali234/Watermark-Removal-System.git
cd Watermark-Removal-System

# Create and activate the conda environment
conda env create -f inpaint.yml
conda activate inpaint
```

### Option 2 — pip

```bash
git clone https://github.com/Ammar-Ali234/Watermark-Removal-System.git
cd Watermark-Removal-System
pip install -r requirements.txt
```

### Option 3 — Docker

```bash
docker build -t watermark-removal .
docker run -p 5000:5000 watermark-removal
```

---

## 📥 Download Pre-trained Model

Download the model files from the link below and place them inside the `model/` directory:

🔗 [Download Model Weights (Google Drive)](https://drive.google.com/drive/folders/1xRV4EdjJuAfsX9pQme6XeoFznKXG0ptJ?usp=sharing)

> **Note:** After downloading, rename `checkpoint.txt` → `checkpoint` (Google Drive sometimes appends `.txt` automatically).

Your directory should look like:

```
model/
├── checkpoint
├── snap-XXXXX.index
├── snap-XXXXX.meta
└── snap-XXXXX.data-00000-of-00001
```

---

## 🧪 Usage

### Single Image

```bash
python main.py --input test1.jpg --output result.png
```

### Batch Processing

```bash
python batch_test.py
```

### Guided Batch Processing (manual mask)

```bash
python guided_batch_test.py
```

### Web App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8504`

---

## 🧠 How It Works

1. **Preprocessing** — The input image is analyzed and a binary mask is generated over the watermark region using `preprocess_image.py`.
2. **Inpainting** — The masked region is passed through a generative deep neural network (`inpaint_model.py`) that reconstructs the missing pixels based on surrounding context.
3. **Postprocessing** — The reconstructed patch is blended back into the original image to produce a seamless result.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3 |
| Deep Learning | TensorFlow |
| Web Framework | Flask |
| Image Processing | OpenCV, NumPy, Pillow |
| Containerization | Docker |
| Environment | Conda |

---

## 📋 Requirements

- Python >=3.6 but <=3.9
- TensorFlow 1.x / 2.x
- OpenCV
- Streamlli
- NumPy
- Pillow

See `requirements.txt` or `inpaint.yml` for the full dependency list.

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## 🙋‍♂️ Author

**Ammar Ali**
- GitHub: [@Ammar-Ali234](https://github.com/Ammar-Ali234)

---

> ⭐ If you find this project useful, please consider giving it a star!
