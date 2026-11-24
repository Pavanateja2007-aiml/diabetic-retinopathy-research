# Diabetic Retinopathy Detection Using Machine Learning

### A Comparative Experimental Study of Five Machine Learning Algorithms for Retinal Fundus Image Classification

<p align="center">

**Artificial Intelligence | Machine Learning | Medical Image Analysis | Healthcare AI**

</p>

---

## 1. Project Overview

Diabetic Retinopathy (DR) is a diabetes-related eye disease that damages the blood vessels of the retina. If not detected and treated early, it can lead to severe visual impairment and blindness.

This research project investigates the application of machine learning algorithms for automated diabetic retinopathy classification using retinal fundus images.

Five supervised machine learning algorithms are implemented and evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest
5. Extreme Gradient Boosting (XGBoost)

The project focuses on image preprocessing, feature extraction, model training, hyperparameter optimization, classification, and performance evaluation.

The primary objective is to investigate the predictive capabilities and limitations of different machine learning algorithms in identifying diabetic retinopathy severity levels.

## 2. Research Objectives

- Develop machine learning models for diabetic retinopathy classification.
- Process retinal fundus images and extract numerical features.
- Implement five supervised classification algorithms.
- Optimize model hyperparameters using cross-validation.
- Evaluate classification performance using standard metrics.
- Analyze class-wise precision, recall, and F1-score.
- Investigate the impact of imbalanced medical image datasets.
- Identify limitations and potential improvements for future research.

## 3. Dataset Description

**Dataset:** Diabetic Retinopathy 2015 – Colored Resized Images

**Platform:** Kaggle

**Dataset:** [Diabetic Retinopathy 2015 – Colored Resized Images](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized)

**Data Type:** Colored retinal fundus images.

### Original Dataset Distribution

The dataset contains 35,126 retinal images distributed across five diabetic retinopathy categories.

| Class | Number of Images |
|---|---:|
| Mild | 2,443 |
| Moderate | 5,292 |
| No_DR | 25,810 |
| Proliferate_DR | 708 |
| Severe | 873 |
| **Total** | **35,126** |

### Classification Categories

| Class | Description |
|---|---|
| No_DR | No diabetic retinopathy |
| Mild | Mild diabetic retinopathy |
| Moderate | Moderate diabetic retinopathy |
| Severe | Severe diabetic retinopathy |
| Proliferate_DR | Proliferative diabetic retinopathy |

### Dataset Imbalance

The dataset is significantly imbalanced. The No_DR category represents approximately 73.47% of the total images, whereas Proliferate_DR and Severe contain considerably fewer samples.

This imbalance can influence model predictions and cause classifiers to favor the majority class.

## 4. Research Methodology

The project follows a structured machine learning workflow.

### Overall Workflow

```text
       Retinal Fundus Image Dataset
                    |
                    v
           Dataset Exploration
                    |
                    v
           Image Preprocessing
                    |
                    v
             Feature Extraction
                    |
                    v
           Feature Matrix Creation
                    |
                    v
            Dataset Preparation
                    |
                    v
        Hyperparameter Optimization
                    |
                    v
               Model Training
                    |
          +---------+---------+
          |         |         |
          v         v         v
         LR        KNN       SVM
          |         |         |
          +---------+---------+
                    |
          +---------+---------+
          |                   |
          v                   v
     Random Forest          XGBoost
          |                   |
          +---------+---------+
                    |
                    v
             Model Evaluation
                    |
                    v
          Comparative Analysis
                    |
                    v
              Research Findings
```

### Feature Extraction

The experiments generate numerical feature representations from retinal fundus images.

The recorded feature dimensions are:

| Algorithm | Number of Features |
|---|---:|
| Logistic Regression | 8,206 |
| KNN | 8,206 |
| Random Forest | 8,206 |
| SVM | 8,206 |
| XGBoost | 8,207 |

### Dataset Preparation

Two experimental dataset configurations are present in the recorded outputs.

**Full-dataset experiments**

Logistic Regression, KNN, and Random Forest use a feature matrix containing 35,126 samples and 8,206 features.

**Sampled-dataset experiments**

SVM and XGBoost use a reduced dataset containing 4,581 samples.

| Class | Processed Images |
|---|---:|
| Mild | 1,000 |
| Moderate | 1,000 |
| No_DR | 1,000 |
| Proliferate_DR | 708 |
| Severe | 873 |
| **Total** | **4,581** |

### Dataset Splitting

The SVM and XGBoost experiments report the following data split:

| Dataset | Samples | Percentage |
|---|---:|---:|
| Training | 3,206 | 69.98% |
| Validation | 687 | 15.00% |
| Testing | 688 | 15.02% |
| **Total** | **4,581** | **100%** |

The full-dataset experiments use separate training, validation, and testing configurations.

## 5. Machine Learning Algorithms

### 5.1 Logistic Regression

Logistic Regression is a supervised classification algorithm that estimates the probability of a sample belonging to a particular class.

**Experimental Configuration:**

- Regularization parameter: C = 10
- Solver: LBFGS
- Cross-validation: 3 folds
- Hyperparameter candidates: 4

**Best Cross-Validation Accuracy:** 55.76%

### 5.2 K-Nearest Neighbors (KNN)

KNN is a distance-based supervised learning algorithm that classifies samples based on their nearest neighboring observations.

**Experimental Configuration:**

- Number of neighbors: 11
- Distance metric: Euclidean
- Weighting: Distance-based

**Best Cross-Validation Accuracy:** 72.91%

### 5.3 Support Vector Machine (SVM)

SVM is a supervised learning algorithm that identifies an optimal decision boundary for separating different classes.

**Experimental Configuration:**

- Regularization parameter: C = 10
- Gamma: Scale
- Cross-validation: 3 folds
- Hyperparameter candidates: 9

**Best Cross-Validation Accuracy:** 30.60%

### 5.4 Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate predictions.

**Experimental Configuration:**

- Number of estimators: 200
- Maximum depth: None
- Minimum samples split: 2
- Minimum samples leaf: 1
- Cross-validation: 3 folds
- Hyperparameter candidates: 24

**Best Cross-Validation Accuracy:** 73.48%

### 5.5 Extreme Gradient Boosting (XGBoost)

XGBoost is an optimized gradient boosting algorithm that builds decision trees sequentially to improve predictive performance.

**Experimental Configuration:**

- XGBoost version: 3.4.1
- Classification type: Multiclass
- Number of classes: 5
- Class balancing weights: Calculated
- Best training iteration: 220

## 6. Technologies and Libraries

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Image Processing | OpenCV, Scikit-image |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Data Visualization | Matplotlib, Seaborn |
| Development Environment | Jupyter Notebook, Google Colab |
| Version Control | Git and GitHub |

## 7. Experimental Results

The following results are based on the experimental output files available in the repository.

### 7.1 Overall Model Performance

| Algorithm | Training Accuracy | Validation Accuracy | Testing Accuracy |
|---|---:|---:|---:|
| Logistic Regression | 99.23% | 52.86% | 53.05% |
| KNN | 100.00% | 73.07% | 72.97% |
| SVM | 100.00% | 31.30% | 27.76% |
| Random Forest | 100.00% | 73.49% | 73.47% |
| XGBoost | 95.88% | 29.84% | 32.56% |

**Note:** These experiments use different dataset configurations. Therefore, the results are descriptive and should not be interpreted as a strictly controlled comparison.

### 7.2 Precision, Recall and F1-Score

| Algorithm | Test Precision | Test Recall | Test F1-Score |
|---|---:|---:|---:|
| Logistic Regression | 59.68% | 53.05% | 55.94% |
| KNN | 61.04% | 72.97% | 62.62% |
| SVM | 28.00% | 27.76% | 27.71% |
| Random Forest | 53.97% | 73.47% | 62.23% |
| XGBoost | 32.58%* | 33.78%* | 32.09%* |

*XGBoost values use the respective macro precision, macro recall, and weighted F1-score reported in its evaluation summary. These metrics use different averaging methods.*

## 8. Detailed Experimental Analysis

### Logistic Regression

- Training accuracy: 99.23%
- Validation accuracy: 52.86%
- Testing accuracy: 53.05%
- Best cross-validation accuracy: 55.76%

The model achieved high training accuracy but considerably lower validation and testing accuracy, indicating a substantial generalization gap.

### K-Nearest Neighbors

- Training accuracy: 100.00%
- Validation accuracy: 73.07%
- Testing accuracy: 72.97%
- Best cross-validation accuracy: 72.91%

KNN achieved relatively high overall testing accuracy. However, its class-wise classification report revealed poor detection of minority classes.

Reported recall:

- Mild: 0%
- Moderate: 2%
- No_DR: 99%
- Proliferate_DR: 0%
- Severe: 0%

This demonstrates that overall accuracy alone does not establish effective five-class classification.

### Support Vector Machine

- Training accuracy: 100.00%
- Validation accuracy: 31.30%
- Testing accuracy: 27.76%
- Best cross-validation accuracy: 30.60%

SVM achieved perfect training accuracy but considerably lower validation and testing performance, suggesting poor generalization under the current experimental configuration.

### Random Forest

- Training accuracy: 100.00%
- Validation accuracy: 73.49%
- Testing accuracy: 73.47%
- Best cross-validation accuracy: 73.48%

Random Forest achieved the highest reported testing accuracy. However, its classification report showed zero recall for Mild, Moderate, Proliferate_DR, and Severe.

Its predictions were heavily influenced by the majority No_DR category.

### XGBoost

- Training accuracy: 95.88%
- Validation accuracy: 29.84%
- Testing accuracy: 32.56%
- Best training iteration: 220

XGBoost demonstrated relatively low classification performance in the current experiment.

**Class-wise test performance:**

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Mild | 0.26 | 0.27 | 0.27 |
| Moderate | 0.22 | 0.18 | 0.20 |
| No_DR | 0.38 | 0.35 | 0.36 |
| Proliferate_DR | 0.43 | 0.52 | 0.47 |
| Severe | 0.34 | 0.36 | 0.35 |

XGBoost showed its highest recall for Proliferate_DR among the five categories in this experiment.

## 9. Research Findings

The experiments reveal several important observations:

1. **Class Imbalance:** The dataset contains significantly more No_DR images than the other categories.
2. **Overfitting:** Several models achieved perfect or nearly perfect training accuracy but considerably lower testing accuracy.
3. **Majority-Class Bias:** KNN and Random Forest achieved approximately 73% testing accuracy, but their class-wise results revealed poor minority-class detection.
4. **Generalization:** Logistic Regression, SVM, and XGBoost also demonstrated limitations in generalizing to unseen test samples.
5. **Evaluation Metrics:** Accuracy alone is insufficient for evaluating medical image classification systems.
6. **Experimental Consistency:** Different dataset sizes and feature configurations require controlled experiments before making definitive model comparisons.

## 10. Repository Structure

```text
diabetic-retinopathy-research/
│
├── .vscode/
│
├── Xgboost/
│   ├── __results___0_1.png
│   ├── __results___0_13.png
│   ├── __results___0_15.png
│   ├── __results___0_16.png
│   ├── xgboost output.txt
│   └── xgboost-diabetic.ipynb
│
├── __pycache__/
│
├── knn/
│   ├── __results___0_3 (1).png
│   ├── __results___0_5.png
│   ├── __results___0_7.png
│   ├── __results___0_9.png
│   ├── knn output.txt
│   └── knn-diabetic.py
│
├── logistic regression/
│   ├── __results___0_1 (1).png
│   ├── __results___0_3.png
│   ├── __results___0_5.png
│   ├── __results___0_7.png
│   ├── log-diabetic.ipynb
│   └── logistic regression output.txt
│
├── random forest/
│   ├── __results___0_1.png
│   ├── __results___0_3 (1).png
│   ├── __results___0_3.png
│   ├── __results___0_5.png
│   ├── __results___0_7.png
│   ├── __results___0_8.png
│   ├── random forest diabetic output.txt
│   └── random-forest-diabetic.ipynb
│
├── svm/
│   ├── download (1).png
│   ├── download (2).png
│   ├── download (3).png
│   ├── download (4).png
│   ├── download.png
│   ├── svm.txt
│   └── svm_diabetic.ipynb
│
└── README.md
```

### Repository Contents

| Directory | Main Implementation | Experimental Output |
|---|---|---|
| Xgboost | xgboost-diabetic.ipynb | xgboost output.txt |
| knn | knn-diabetic.py | knn output.txt |
| logistic regression | log-diabetic.ipynb | logistic regression output.txt |
| random forest | random-forest-diabetic.ipynb | random forest diabetic output.txt |
| svm | svm_diabetic.ipynb | svm.txt |

The PNG files contain saved visualizations associated with the corresponding experiments.

## 11. Installation and Execution

### Prerequisites

- Python 3.x
- Jupyter Notebook or Google Colab
- Git
- Required Python libraries

### Step 1: Clone the Repository

```bash
git clone https://github.com/Pavanateja2007-aiml/diabetic-retinopathy-research.git
```

### Step 2: Navigate to the Project

```bash
cd diabetic-retinopathy-research
```

### Step 3: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scikit-image opencv-python xgboost
```

### Step 4: Configure the Dataset

Download the dataset from Kaggle and configure the dataset directory according to the paths used in the respective implementations.

### Step 5: Execute the Models

Open the respective notebooks or Python scripts and execute them in the appropriate environment.

**Note:** Dataset paths and execution settings may need modification when running outside the original development environment.

## 12. Limitations

- Significant class imbalance affects minority-class classification.
- Several models demonstrate a considerable training and testing performance gap.
- Some models favor the majority No_DR category.
- Experiments use different dataset configurations and feature dimensions.
- Independent external validation has not been demonstrated in the supplied outputs.
- Patient-level data leakage has not yet been ruled out.
- The experimental results do not establish clinical diagnostic reliability.

## 13. Future Scope

Potential research improvements include:

- Implement Convolutional Neural Networks (CNNs) for retinal image classification.
- Explore transfer learning using ResNet, EfficientNet, and DenseNet.
- Investigate advanced retinal image preprocessing techniques.
- Apply class balancing strategies and data augmentation.
- Perform systematic hyperparameter optimization.
- Introduce Explainable AI techniques such as Grad-CAM.
- Evaluate models using patient-level splits.
- Validate performance on independent retinal image datasets.
- Develop a research demonstration application using Flask or FastAPI.

## 14. Conclusion

This research investigates five machine learning algorithms for diabetic retinopathy classification using retinal fundus images.

The experimental findings demonstrate that achieving high overall accuracy does not necessarily guarantee effective classification across all disease categories.

Although KNN and Random Forest achieved approximately 73% testing accuracy, their class-wise evaluation revealed significant limitations in identifying minority diabetic retinopathy categories.

The study highlights the importance of class-balanced evaluation, reliable feature representation, appropriate validation strategies, and model generalization in developing machine learning systems for healthcare applications.

Future research involving deep learning, improved preprocessing, robust validation, and explainable AI may help address the limitations identified in these experiments.

## 15. Medical Disclaimer

This project is intended exclusively for educational and research purposes. It is not a clinically validated medical diagnostic system.

Predictions generated by the experimental models must not be used as standalone tools for medical diagnosis or treatment decisions. Professional ophthalmological evaluation is essential for diabetic retinopathy diagnosis.

## 16. Author

**Vemuri Venkata Satya Markandeya Pavanateja**

B.Sc. Artificial Intelligence  
Department of Computer Science

**GitHub:** [Pavanateja2007-aiml](https://github.com/Pavanateja2007-aiml)

**Project Repository:** [Diabetic Retinopathy Research](https://github.com/Pavanateja2007-aiml/diabetic-retinopathy-research)

---

<p align="center">

**Exploring Artificial Intelligence for Healthcare Research**

⭐ If you find this project useful, consider giving the repository a star!

</p>
