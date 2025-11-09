Here’s a concise **README** you can include in your Colab notebook or GitHub repo to explain your TP3 exercises, the setup, and how to run them:

---

# TP3 – Image and Video Processing Exercises

This notebook contains exercises on **image processing, video capture, object detection, and face recognition** using Python libraries such as OpenCV, Detectron2, Dlib, Face Recognition, and MediaPipe.

---

## **Notes**

* Exercises 1, 3, 4, 5 run well on Colab with shared images.
* Exercise 2 requires **local machine** execution due to webcam restrictions in Colab.
* Adjust thresholds for face detection/recognition as needed.
* For multiple faces, compare each detected face individually.
---
## **Google Drive and Notebook Links**

* **Folder with images :** [TP3 Shared Drive](https://drive.google.com/drive/folders/1qki7VJa8vt4dnj6Gn23ZjT5kC243IFlE?usp=sharing)
* **Colab Notebook:** [TP3.ipynb](https://colab.research.google.com/drive/1U9F2880F1UsCTNXk8h-ZBDDBKwRv3PE-?usp=drive_link)

---

## **Setup Instructions**

1. Mount Google Drive to access files:

```python
from google.colab import drive
drive.mount('/content/MyDrive')
```

2. Change to your working directory:

```python
import os
data_path = "/content/MyDrive/MyDrive/Colab_Notebooks/Khmlichi/TP3"
os.chdir(data_path)
```

3. Install necessary libraries:

```bash
!pip install mediapipe==0.10.21 opencv-python-headless==4.11.0.46 matplotlib numpy==1.26.4
!pip install -U git+https://github.com/facebookresearch/detectron2.git
!pip install dlib face_recognition
```

> **Note:** Some exercises require a desktop environment (webcam access), and face recognition may require Python ≤ 3.10 for compatibility in Colab.

---

## **Exercise Overview**

### **Exercice 1 – Image Analysis**

* Load an image.
* Display it with OpenCV.
* Convert it to grayscale.
* Apply Canny edge detection.
* **Libraries used:** OpenCV, Matplotlib.

### **Exercice 2 – Video Capture / Webcam**

* Display live webcam video.
* Apply a grayscale filter in real-time.
* Close on pressing `q`.
* **Important:** Only runs on **local machine** (webcam access not available in Colab).

### **Exercice 3 – Object Detection (Detectron2)**

* Load an image with multiple objects.
* Detect and segment objects using **Mask R-CNN**.
* Display image with colored masks and labels.
* **Libraries used:** Detectron2, OpenCV.

### **Exercice 4 – Face Detection (Dlib)**

* Load an image with people.
* Detect all faces and draw green rectangles around them.
* **Libraries used:** Dlib, OpenCV.

### **Exercice 5 – Face Recognition (Face_Recognition / MediaPipe alternative)**

* Load reference and test images.
* Compare faces and indicate if it’s the same person.
* **Colab-friendly alternative:** MediaPipe embeddings + cosine similarity.
* **Libraries used:** Face Recognition (or MediaPipe + OpenCV).

---

## **MediaPipe-based Face Comparison (Colab-friendly)**

```python
import cv2, mediapipe as mp, numpy as np, matplotlib.pyplot as plt

mp_face_detection = mp.solutions.face_detection

def get_face_embedding(image_path):
    image = cv2.imread(image_path)
    if image is None: return None, None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(image_rgb)
        if not results.detections: return None, image_rgb
        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = image_rgb.shape
        x1, y1 = max(int(bbox.xmin*w),0), max(int(bbox.ymin*h),0)
        x2, y2 = min(x1+int(bbox.width*w), w), min(y1+int(bbox.height*h), h)
        face_crop = cv2.resize(image_rgb[y1:y2,x1:x2], (128,128))
        embedding = face_crop.flatten().astype(np.float32)
        embedding /= np.linalg.norm(embedding)+1e-10
        return embedding, image_rgb

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

# Example usage
ref_img = "Jim_Carrey_2008.jpg"
test_img = "Jim_Carrey_2010.jpg"
enc_ref, img_ref = get_face_embedding(ref_img)
enc_test, img_test = get_face_embedding(test_img)
if enc_ref is not None and enc_test is not None:
    similarity = cosine_similarity(enc_ref, enc_test)
    match = similarity>0.6
    print(f"Same person? {match} (similarity={similarity:.2f})")
```

---

## **Notes**

* Exercises 1, 3, 4, 5 run well on Colab with shared images.
* Exercise 2 requires **local machine** execution due to webcam restrictions in Colab.
* Adjust thresholds for face detection/recognition as needed.
* For multiple faces, compare each detected face individually.

---