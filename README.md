# Virtual Try-On for Glasses & Accessories

A real-time Computer Vision application that detects your face using a webcam and overlays virtual accessories — sunglasses, hats, and masks — directly onto your face using facial landmark detection.

Built with MediaPipe Face Mesh and OpenCV as part of a Computer Vision course project.

---

## Demo

| Sunglasses | Party Hat | Medical Mask |
|------------|-----------|--------------|
| Detected via eye landmarks | Anchored to forehead point | Placed between nose and chin |

Run the app and press D to cycle through accessories live.

---

## How It Works

1. **Face Detection** — MediaPipe's Face Mesh model detects 468 facial landmarks in real time from webcam input.
2. **Landmark Anchoring** — Specific landmark indices are used as anchor points:
   - Glasses: outer eye corners (landmarks 33, 263)
   - Hats: forehead point (landmark 10)
   - Masks: nose tip and chin (landmarks 1, 152)
3. **Image Overlay** — PNG accessories with alpha transparency are resized and blended onto the frame using per-pixel alpha compositing.
4. **Mirrored View** — The frame is horizontally flipped for a natural mirror feel.

---

## Project Structure

```
virtual_tryon/
│
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
└── accessories/            # Auto-generated PNG assets (or replace with your own)
    ├── sunglasses.png
    ├── round_glasses.png
    ├── party_hat.png
    ├── cowboy_hat.png
    ├── medical_mask.png
    └── ninja_mask.png
```

---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- A working webcam
- Git

### Step 1 - Clone the repository
```bash
git clone https://github.com/Sia2005/virtual-tryon-cv.git
cd virtual-tryon-cv
```

### Step 2 - Create a virtual environment (recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3 - Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 - Run the app
```bash
python app.py
```

The app will auto-generate placeholder accessories on the first run if none are found. You can replace them with your own PNG files with transparency (RGBA).

---

## Controls

| Key | Action |
|-----|--------|
| D | Next accessory |
| A | Previous accessory |
| L | Toggle facial landmark overlay |
| Q or ESC | Quit |

---

## Using Custom Accessories

You can replace any file in the accessories/ folder with your own PNG image:
- Must be PNG format with transparency (RGBA / 4 channels)
- Glasses: horizontal orientation works best (wider than tall)
- Hats: should have the brim at the bottom of the image
- Masks: roughly square works well

The app handles resizing automatically based on facial proportions.

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.8.0 or higher | Frame capture, image processing, display |
| mediapipe | 0.10.0 or higher | Face mesh and 468-point landmark detection |
| numpy | 1.24.0 or higher | Array operations for alpha blending |
| keyboard | latest | Global keypress detection |

---

## CV Concepts Used

- **Face Mesh Detection** — MediaPipe's neural network detects a 3D face mesh with 468 landmarks
- **Facial Landmark Alignment** — Anchor points are used to correctly size and position accessories
- **Alpha Compositing** — Per-pixel blending using the formula: output = fg * alpha + bg * (1 - alpha)
- **Affine Scaling** — Accessories are dynamically scaled relative to inter-eye or face width
- **Real-time Video Processing** — Frame-by-frame processing pipeline with OpenCV VideoCapture

---

## Troubleshooting

**Webcam not opening?**
- Check that no other app is using the camera
- Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1) in app.py

**Accessories misaligned?**
- Ensure good lighting so landmarks are detected accurately
- Face the camera directly; extreme angles reduce accuracy

**mediapipe install fails on Apple Silicon?**
```bash
pip install mediapipe-silicon
```

**Keys not responding?**
- Run the script as Administrator (required for the keyboard library on Windows)

---

## License

This project is submitted as a course project for academic evaluation purposes.

---

## Author

Sia — GitHub: Sia2005
Course: Computer Vision
Submission: March 2026
