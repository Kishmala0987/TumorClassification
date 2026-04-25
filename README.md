# 🧠 Brain Tumor Classification using Deep Learning

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)
![Type](https://img.shields.io/badge/Project-Medical%20AI-purple)
![Demo](https://img.shields.io/badge/Live-Demo-orange)

---

## 🌐 Live Demo

🚀 Try the model here: **[Hugging Face Space](https://huggingface.co/spaces/newbietoken/TumorClassification)**

> Upload an MRI scan and get instant predictions with visual explanations (Grad-CAM++ & LIME).

---

## 📌 Overview

This project presents an end-to-end deep learning system for automated classification of brain MRI scans into tumor categories. It combines **transfer learning-based CNNs** with **explainable AI techniques** to produce both accurate and interpretable predictions.

The pipeline is designed to simulate real-world medical AI workflows, including preprocessing, inference, and post-hoc explanation generation.

---

## 🧬 Classes

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor

---

## 🏗️ Model Architecture

* **Backbone:** CNN (VGG16 / ResNet / EfficientNet)
* **Input:** 224 × 224 MRI images
* **Output:** Multi-class classification
* **Loss:** Cross-Entropy
* **Optimizer:** Adam

  <img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/07f00bec-0f52-4129-99ee-8cc613e5cfab" />


📌 Transfer learning enables efficient training and strong performance even with limited medical datasets.

---

## 🔍 Explainability (Grad-CAM++ & LIME)

To improve model transparency, the system integrates:

* **Grad-CAM++** → Highlights spatial regions influencing predictions
* **LIME** → Provides local, interpretable feature importance

### Why it matters:

* Helps validate whether the model focuses on tumor regions
* Builds trust in AI-assisted diagnosis
* Supports research in interpretable medical AI

---

## 📊 Results

* **Accuracy:** 96%
* **Macro F1-Score:** 90%
* Robust performance across all tumor classes
* Strong generalization on unseen MRI data

---

## 🖼️ Sample Outputs

### Explainability AI

## GRAD CAM++

<img width="970" height="503" alt="image" src="https://github.com/user-attachments/assets/adca30a7-da80-424f-a69b-ba51c4081d22" />

---

## LIME

<img width="963" height="513" alt="image" src="https://github.com/user-attachments/assets/ee4ce54a-2680-4362-9960-6cfc056e2981" />

---

## Overlay: LIME, Grad-CAM, and Predictions

<img width="2663" height="1588" alt="image" src="https://github.com/user-attachments/assets/062b0bc9-2eb4-41d2-bee8-47fd786942de" />

---

## ⚙️ Tech Stack

* Python
* PyTorch / TensorFlow
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/tumor-classification.git
cd tumor-classification
pip install -r requirements.txt
```

---

## 📌 Key Features

✔ Multi-class brain tumor classification
✔ Transfer learning-based CNN models
✔ Integrated Grad-CAM++ and LIME explainability
✔ Interactive deployment via Hugging Face
✔ Clean, modular, and reproducible pipeline

---

## 🧠 Future Improvements

* Extend to 3D MRI volumetric data
* Incorporate attention mechanisms
* Clinical dataset validation
* Real-time hospital integration systems

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only** and should not be used for real clinical diagnosis.

---

## 👩‍💻 Author

**Kishmala Khan**
AI/ML Research Enthusiast | Medical AI | Explainable AI

---
