# **Video Analysis and Object Detection Framework**

This repository contains a Python-based framework for processing video files to extract frames, detect keyframes, and perform object detection using YOLO. Below is a detailed analysis of the implemented methods, their performance, and references to relevant research papers.

---

## **Features**

1. Frame Extraction
2. Uniform Sampling
3. Scene Change Detection
4. Keyframe Clustering
5. Object Detection with YOLOv8
6. Frame Visualization

---

## **1. Frame Extraction**

### **Methodology:**

- Extracts all frames from a given video using OpenCV's `VideoCapture`.
- Frames are saved as individual images in the specified output directory.

### **Performance:**

- Efficient for videos with a manageable frame count.
- For high-frame-rate videos, the storage requirement increases significantly.

### **Applications:**

- Useful for preprocessing videos for tasks like summarization or content-based retrieval.

### **Citations:**

- OpenCV Documentation: [https://opencv.org/](https://opencv.org/)

---

## **2. Uniform Sampling**

### **Methodology:**

- Selects frames at uniform intervals from the extracted frames.
- The interval is calculated as `len(frames) // num_samples`.

### **Performance:**

- Ensures coverage across the entire video without processing all frames.
- May miss critical frames if changes occur between sampled frames.

### **Applications:**

- Effective for reducing computational load while maintaining temporal diversity.

---

## **3. Scene Change Detection**

### **Methodology:**

- Uses histogram comparisons to detect significant visual changes.
- Compares the correlation of successive frame histograms. Frames with a correlation below a threshold are marked as keyframes.

### **Performance:**

- **Strengths:** Detects abrupt changes efficiently.
- **Limitations:** Struggles with gradual transitions like fading or panning.

### **Applications:**

- Keyframe extraction for video summarization.

### **Citations:**

- Histogram Comparison: ["A new histogram similarity measure for robust image registration"](https://www.sciencedirect.com/science/article/pii/S089561119600045X)

---

## **4. Keyframe Clustering**

### **Methodology:**

- Extracts visual features by resizing frames and flattening pixel data.
- Reduces feature dimensions using PCA.
- Groups frames into clusters using K-Means and selects the first frame of each cluster as representative.

### **Performance:**

- **Strengths:** Captures diverse content efficiently.
- **Limitations:** Computationally expensive for high-dimensional data.

### **Applications:**

- Selecting representative frames for video summarization or content indexing.

### **Citations:**

- PCA: ["Principal Component Analysis"](https://doi.org/10.1080/10618600.1992.10475879)
- K-Means: ["K-means clustering"](https://doi.org/10.1016/0377-0427\(84\)90080-3)

---

## **5. Object Detection with YOLOv8**

### **Methodology:**

- Uses the YOLOv8 model for real-time object detection.
- Bounding boxes are drawn around detected objects, excluding those labeled as "person" for specific use cases.

### **Performance:**

- **Strengths:** Fast and accurate for most objects.
- **Limitations:** Accuracy depends on the quality of the pre-trained YOLO model.

### **Applications:**

- Detecting objects in surveillance videos or autonomous driving.

### **Citations:**

- YOLOv8: [Ultralytics Documentation](https://docs.ultralytics.com)
- Original YOLO Paper: ["You Only Look Once: Unified, Real-Time Object Detection"](https://arxiv.org/abs/1506.02640)

---

## **6. Visualization**

### **Methodology:**

- Displays frames with bounding boxes using Google Colab's `cv2_imshow`.

### **Performance:**

- Simplifies the validation of detection results.
- Limited to environments with image display support (e.g., Google Colab).

---

## **Comparison of Methods**

| **Method**              | **Strengths**                       | **Limitations**                              |
| ----------------------- | ----------------------------------- | -------------------------------------------- |
| Frame Extraction        | High fidelity, preserves all frames | High storage and processing cost             |
| Uniform Sampling        | Reduces processing load             | May miss critical moments                    |
| Scene Change Detection  | Captures abrupt changes             | Struggles with gradual transitions           |
| Keyframe Clustering     | Ensures diversity in content        | Computationally expensive for large datasets |
| YOLOv8 Object Detection | Real-time and accurate              | Requires a high-quality pre-trained model    |

---

## **Future Enhancements**

1. Incorporate gradual scene change detection (e.g., using edge detection or optical flow).
2. Use deep learning-based keyframe extraction for better performance on complex videos.
3. Integrate additional YOLO models for specialized object detection tasks.

---

## **References**

- OpenCV: [https://opencv.org/](https://opencv.org/)
- PCA Research: ["Principal Component Analysis"](https://doi.org/10.1080/10618600.1992.10475879)
- K-Means Clustering: ["K-means clustering"](https://doi.org/10.1016/0377-0427\(84\)90080-3)
- YOLOv8 Documentation: [https://docs.ultralytics.com](https://docs.ultralytics.com)
- Original YOLO Paper: ["You Only Look Once"](https://arxiv.org/abs/1506.02640)
- Histogram Comparison: ["A new histogram similarity measure for robust image registration"](https://www.sciencedirect.com/science/article/pii/S089561119600045X)
- For more Advanced Implementation: 
  - ([https://openaccess.thecvf.com/content\_WACV\_2020/papers/Ren\_Best\_Frame\_Selection\_in\_a\_Short\_Video\_WACV\_2020\_paper.pdf](https://openaccess.thecvf.com/content_WACV_2020/papers/Ren_Best_Frame_Selection_in_a_Short_Video_WACV_2020_paper.pdf))
  - Localisation ([https://arxiv.org/pdf/2004.12276](https://arxiv.org/pdf/2004.12276)) 

---
