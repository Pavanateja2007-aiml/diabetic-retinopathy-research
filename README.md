# Diabetic Retinopathy Detection Using Machine Learning

<p align="center">
  <b>A Comparative Machine Learning Research Project for Diabetic Retinopathy Classification</b>
</p>

## 📌 Project Overview

Diabetic Retinopathy (DR) is a diabetes-related eye disease that affects the blood vessels of the retina and can lead to vision impairment or blindness if not detected early.

This project focuses on developing and evaluating machine learning models to classify diabetic retinopathy using retinal fundus images.

The primary objective is to investigate different machine learning algorithms, compare their classification performance, and identify suitable approaches for automated diabetic retinopathy detection.

## 🎯 Research Objectives

- Develop machine learning models for diabetic retinopathy classification.
- Perform image preprocessing and feature extraction.
- Implement multiple classification algorithms.
- Evaluate model performance using standard classification metrics.
- Compare the performance of different machine learning approaches.
- Explore the potential of AI-assisted retinal disease screening.

## 📂 Project Structure

```text
diabetic-retinopathy-research/
│
├── Xgboost/
│
├── knn/
│
├── logistic regression/
│
├── random forest/
│
├── svm/
│
├── .vscode/
│
└── README.md
```

## 🗂️ Dataset

The project uses a diabetic retinopathy retinal fundus image dataset.

**Dataset:** Diabetic Retinopathy 2015 - Colored Resized Images

**Dataset Source:** Kaggle

[Access Dataset](https://www.kaggle.com/)

The dataset contains retinal fundus images used for developing and evaluating machine learning classification models.

Dataset preparation includes:
- Loading retinal fundus images.
- Image resizing and preprocessing.
- Feature extraction.
- Label encoding and class preparation.
- Preparing data for model training and evaluation.

## 🧠 Machine Learning Algorithms

This research implements and investigates five machine learning algorithms.

### 1. Logistic Regression
A supervised classification algorithm used as a baseline model to establish an initial classification performance.

### 2. K-Nearest Neighbors (KNN)
A distance-based classification algorithm that predicts the class of an image based on its nearest neighboring samples.

### 3. Support Vector Machine (SVM)
A supervised learning algorithm that identifies optimal decision boundaries for separating different classes.

### 4. Random Forest
An ensemble learning algorithm that combines multiple decision trees to improve classification performance and reduce overfitting.

### 5. XGBoost
An optimized gradient boosting algorithm that builds sequential decision trees to improve predictive performance.

## ⚙️ Technologies and Libraries

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Development Environment | Jupyter Notebook, Google Colab |
| Data Processing | NumPy, Pandas |
| Image Processing | OpenCV, Scikit-image |
| Machine Learning | Scikit-learn |
| Boosting | XGBoost |
| Visualization | Matplotlib, Seaborn |
| Model Evaluation | Scikit-learn Metrics |

## 📊 Model Evaluation

The implemented models are evaluated using standard classification metrics:

- **Accuracy:** Measures the overall correctness of predictions.
- **Precision:** Measures how many predicted positive samples are actually positive.
- **Recall (Sensitivity):** Measures how many actual positive samples are correctly identified.
- **F1-Score:** Calculates the harmonic mean of precision and recall.
- **Confusion Matrix:** Visualizes correct and incorrect classifications.

These metrics help analyze the strengths and weaknesses of each classification algorithm.

## 🔬 Research Methodology

The overall research workflow follows these stages:

1. Dataset Collection
2. Image Loading and Exploration
3. Image Preprocessing
4. Feature Extraction
5. Data Preparation
6. Model Training
7. Model Prediction
8. Performance Evaluation
9. Comparative Analysis

## 🚀 Installation and Execution

### Prerequisites

- Python 3.x
- Jupyter Notebook or Google Colab
- Basic knowledge of Python and Machine Learning

### Step 1: Clone the Repository

```bash
git clone https://github.com/Pavanateja2007-aiml/diabetic-retinopathy-research.git
```

### Step 2: Navigate to the Project

```bash
cd diabetic-retinopathy-research
```

### Step 3: Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scikit-image opencv-python xgboost
```

### Step 4: Open the Project

Open the respective algorithm folders and execute the notebooks or Python scripts in your preferred development environment.

**Note:** Download and configure the dataset separately before executing the models.

## 📈 Experimental Results

The performance of the implemented algorithms should be compared using the following metrics.

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | To be updated | To be updated | To be updated | To be updated |
| KNN | To be updated | To be updated | To be updated | To be updated |
| SVM | To be updated | To be updated | To be updated | To be updated |
| Random Forest | To be updated | To be updated | To be updated | To be updated |
| XGBoost | To be updated | To be updated | To be updated | To be updated |

*Actual experimental results will be added after completing model evaluation. No performance values are assumed.*

## ⚠️ Limitations

- Model performance depends on dataset quality and diversity.
- Class imbalance may affect classification performance.
- Image quality and preprocessing choices can influence predictions.
- External validation is necessary to assess generalization.
- This research project is not a substitute for professional ophthalmological diagnosis.

## 🔮 Future Enhancements

- Explore Convolutional Neural Networks (CNNs) for retinal image classification.
- Investigate transfer learning using pretrained deep learning architectures.
- Implement explainable AI techniques such as Grad-CAM.
- Improve model generalization through robust validation.
- Develop a web-based diabetic retinopathy screening application.
- Explore deployment of trained models using Flask or FastAPI.

## 👨‍💻 Author

**Vemuri Venkata Satya Markandeya Pavanateja**

B.Sc. Artificial Intelligence Student | AI & Machine Learning Researcher

- **GitHub:** [Pavanateja2007-aiml](https://github.com/Pavanateja2007-aiml)
- **Repository:** [Diabetic Retinopathy Research](https://github.com/Pavanateja2007-aiml/diabetic-retinopathy-research)

## 📜 Disclaimer

This project is developed for educational and research purposes. The implemented machine learning models are experimental and should not be used as standalone medical diagnostic systems.

---

⭐ If you find this research useful, consider starring the repository!
