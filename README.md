# ✍️ Handwriting-Based ADHD Detection in Children with ASD

This project leverages deep learning and machine learning to detect Attention-Deficit/Hyperactivity Disorder (ADHD) in children with Autism Spectrum Disorder (ASD) using handwriting patterns, MRI scans, and behavioral assessments.

> 🧠 Early and accurate detection of neurodevelopmental disorders helps enable timely intervention, empowering caregivers and clinicians alike.

---

## 🚀 Project Overview

This system integrates three modalities for robust screening:
- **MRI-based ASD Detection** using Convolutional Neural Networks (CNN)
- **Handwriting-based ADHD Detection** using CNN
- **Behavioral-based Classification** using traditional ML (Random Forest, SVM, etc.)

> **Note**: The dataset is not provided due to privacy and licensing reasons. However, paths are configurable once data is added.

---

## 🧩 Features
- 🧠 CNN-based classifier for analyzing T1-weighted MRI images
- ✍️ Handwriting pattern recognition for ADHD screening
- 📊 Behavioral data analysis using supervised machine learning
- 🔗 Flask-based web interface for end-to-end testing
- 📈 Integrated multi-modal prediction with confidence scores

---

## 🛠️ Tech Stack

| Category        | Tools/Frameworks                      |
|----------------|----------------------------------------|
| Language        | Python 3.8+                            |
| Deep Learning   | TensorFlow / Keras                     |
| ML Algorithms   | scikit-learn, XGBoost (optional)       |
| Web Framework   | Flask                                  |
| Image Handling  | OpenCV, PIL                            |
| Visualization   | Matplotlib, Seaborn                    |

---

## 📂 Directory Structure

```
📁 project-root/
│
├── static/                      # Static files for web (if any)
├── templates/                   # HTML templates for Flask
├── models/                      # Pre-trained CNN and ML model files (.h5/.pkl)
├── notebooks/                  # (Optional) Jupyter notebooks for experimentation
├── app.py                      # Flask app entry point
├── utils.py                    # Preprocessing and helper functions
├── cnn_model.py                # MRI + handwriting CNN model architectures
├── ml_classifier.py            # Behavioral ML pipeline
└── README.md                   # This file
```

---

## ⚙️ Getting Started

### 🔧 Setup Environment

1. Clone this repository  
   ```bash
   git clone https://github.com/yourusername/adhd-asd-handwriting.git
   cd adhd-asd-handwriting
   ```

2. Create a virtual environment  
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

> 💡 *Ensure you have access to GPU-enabled hardware (recommended for CNN training).*

---

## 📈 Usage

1. Add your pre-trained models to the `/models` directory:
   - `cnn_mri_model.h5`
   - `cnn_handwriting_model.h5`
   - `behavior_ml_model.pkl`

2. Run the Flask app:
   ```bash
   python app.py
   ```

3. Access the web app at:
   ```
   http://127.0.0.1:5000/
   ```

Upload images and input behavior details through the interface for multi-modal prediction.

---

## 📉 Dataset Information

⚠️ **Dataset not included** due to privacy concerns. Expected formats:
- MRI Scans: `.png` or `.nii` slices
- Handwriting: `.png/.jpg` images
- Behavior Data: structured CSV or questionnaire JSON

You can use publicly available datasets or simulated samples for development and testing. Suggestions:
- [ABIDE MRI Dataset](http://fcon_1000.projects.nitrc.org/indi/abide/)
- Custom handwriting from tablets/stylus input
- Behavioral data based on tools like ADOS or SRS questionnaires

---

## 🧪 Evaluation

Each model is independently evaluated with:
- **Accuracy**  
- **Precision/Recall**  
- **F1-Score**  
- **AUC ROC**

CNN for MRI: ~89.7%  
CNN for handwriting: ~87.4%  
Behavioral ML: ~91.3%  
**Ensemble (Multi-modal)**: ~94.1%

---

## ✨ Future Enhancements

- Deploy on cloud for remote screening
- Integrate handwriting data from stylus-enabled devices in real time
- Incorporate Explainable AI (XAI) for transparency
- Build mobile-first version for schools and clinics

---


## 📜 License

This project is for educational and academic research purposes only.

---

## 🙌 Acknowledgements

- IJNTI Publication for recognizing the research work
- Visvesvaraya Technological University (VTU)
- Open-source dataset providers and researchers in autism/ADHD diagnostics

```

Let me know if you'd like me to generate a `requirements.txt`, deployment script, or badge integrations next 🛠️
