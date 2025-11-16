# Mental-Burnout-Detection-System

**Tagline:** A multi-modal system using facial, vocal, and questionnaire-based analysis to detect early signs of mental burnout.

***

## 1. Introduction

Mental burnout is a growing concern among students, professionals, and healthcare workers. It leads to decreased productivity, poor mental health, and long-term health issues.

Our project addresses this by creating a multi-modal burnout detection system that integrates:

* **Questionnaires** for self-reported symptoms.
* **Facial Analysis** (eye and mouth fatigue indicators + emotion recognition).
* **Voice Analysis** (sentiment, transcription).
* **outcome**(it will also provide a personalized burnout score and coping suggestions based on the analysis.)

This holistic approach enables more accurate, real-time detection of burnout signs compared to single-modality systems.

***

## 2. Features

* Interactive web-based questionnaire for self-assessment.
* Real-time facial analysis using **EAR (Eye Aspect Ratio)**, **MAR (Mouth Aspect Ratio)**, and emotion detection.
* Voice-based analysis for sentiment .
* Speech-to-text transcription for further semantic understanding.
*  Provides a personalized burnout score and coping suggestions based on the analysis.
* **Live Demo:** [Check out the live demo on Hugging Face Spaces](https://huggingface.co/spaces/project-exhibition/Burnout-detection).

***

## 3. Technical Details and Methodology
### Questionnaire

- Collects user-reported stress and fatigue levels.

- Based on validated clinical burnout scales ([Burnout scale](https://link.springer.com/article/10.1007/s11606-014-3112-6)).


### Facial Analysis

- **EAR (Eye Aspect Ratio):** Detects eye closure, drowsiness, and fatigue .([source]( https://www.mdpi.com/1424-8220/24/17/5683)).

- **MAR (Mouth Aspect Ratio):** Captures yawning/fatigue signals.

- **Emotion Detection:** Uses DeepFace for classifying facial emotions (happy, sad, stressed, angry, etc.) ([source]( https://arxiv.org/abs/2504.03010)).


 ### Voice Analysis

- **Speech-to-Text:** Converts audio input into text for further sentiment analysis.

- **Sentiment Analysis:** Implements VADER sentiment scoring on transcripts.
### Dataset: Description and Collection

The dataset for this study was collected using a custom-built, multi-modal data acquisition tool designed to measure psychological burnout indicators. The tool combines self-reported survey data with physiological and emotional data captured through a webcam and microphone, with a total of **326 samples** collected.

The data collection was a sequential, three-part process:

1.  **Self-Reported Burnout Questionnaire:** Participants first completed a six-question survey to assess various facets of academic and personal burnout using a Likert scale.
2.  **Facial and Emotional Analysis:** Participants engaged in a 20-second facial analysis session using a webcam. The tool measured key facial indicators like **Eye Aspect Ratio (EAR)**, **Mouth Aspect Ratio (MAR)**, and classified emotions into positive, neutral, and negative categories.
3.  **Voice and Sentiment Analysis:** Participants provided a  voice sample, which was processed to extract features and perform sentiment analysis. The audio was converted into text using a speech recognition API, and **VADER Sentiment Scores** were generated.

### Data Preprocessing and Preparation

The data preprocessing phase refined the raw dataset for machine learning by addressing quality issues, standardizing features, and mitigating class imbalance.

* **Data Cleaning and Encoding:** Qualitative data, such as burnout labels and frequency responses, were mapped to a numerical scale of 0-4. A specific subset of 20 data points was then removed, and any rows with missing values were dropped.
* **Binning and Consistency Filtering:** The continuous `Sentiment_Comp` feature was partitioned using both equal-frequency (pd.qcut) and equal-width (pd.cut) methods. Only data points where both methods yielded the same bin label were kept to isolate the most reliable data.
* **Final Model Preparation:** The dataset was split into an 80% training and 20% testing set using a **stratified split** to maintain class proportions. Numerical features were then **standardized** using a StandardScaler, and the **SMOTE** technique was applied to the training data to synthetically balance class distribution.

***

## 4. Model Performance

* **CatBoost (Accuracy: 0.78):** This model emerged as the most accurate single model, outperforming all other classifiers due to its specialized approach to handling categorical features and unique training methods for preventing overfitting.
* **CatBoost + SVM Stacking (Accuracy: 0.74):** This ensemble paired the two highest-performing individual models, resulting in the highest macro average F1-score and confirming that combining the strongest individual classifiers yields the most robust final model.
* **Random Forest + CatBoost Stacking (Accuracy: 0.74):** This hybrid ensemble combined a bagging model with a boosting model, leveraging their different strengths to achieve a notable improvement.
* **Support Vector Machine (SVM) (Accuracy: 0.72):** SVM excelled by using a non-linear kernel to find an optimal hyperplane in a high-dimensional space, validating the presence of complex, non-linear patterns in the data.
* **Random Forest (Accuracy: 0.69):** This bagging-based ensemble model delivered solid performance, providing a significant accuracy boost over the baseline and proving its resistance to overfitting and ability to handle non-linear relationships.
* **Logistic Regression (Accuracy: 0.54):** Serving as the baseline, this linear model's performance was limited by its inability to capture the complex, non-linear relationships in the data. Its modest accuracy indicated that a more sophisticated approach was necessary.

***

## 5. Getting Started

### Prerequisites

* **Hardware:** Webcam & Microphone.
* **Software:** Python 3.10, pip.

### Installation

```bash
# Clone repository
git clone [https://github.com/your-username/mental-burnout-detector.git](https://github.com/your-username/mental-burnout-detector.git)
cd mental-burnout-detector

# Install dependencies
pip install -r requirements.txt
python app.py
```

### 6. File Structure
```bash
.
├── app.py                  # Main Flask/Streamlit application                
├── deepface.py             # Emotion detection module               
├── ear_mar_calculation.py  # EAR & MAR Fatigue detection   
├── voice.py                # Voice analysis   
├── requirements.txt        # Dependencies   
├── literature.pdf          # Research references
├── data - data (6).csv     # Raw dataset
├── model.joblib            # Trained CatBoost model
├── runtime.txt             # Python runtime specification
├── index.html              # Website frontpage
├── next.html               # Website second page
├── style.css               # Website frontpage styling
├── style2.css              # Website second page styling
├── scaler.joblib           # StandardScaler object
└── README.md               # Project Documentation
```
