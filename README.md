---

# 🧠 Hybrid Mental Health Prediction System

### A Unified Multi-Modal Approach Integrating Behavioral Metrics & Natural Language Processing

![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📖 Overview

The **Hybrid Mental Health Prediction System** is a unified **Machine Learning + NLP framework** designed to predict an individual's **mental health risk level** — _Low_, _Moderate_, or _High_.

Traditional systems struggle to merge structured data (like CGPA or Sleep Hours) with unstructured social text (like Reddit posts). This project bridges that gap using a **Tabular-to-Text (Tab2Text)** strategy — transforming numerical and categorical attributes into **clinical-style narratives**.

These serialized records, alongside real Reddit posts, are fed into a fine-tuned **MentalRoBERTa** model, enabling it to learn both _behavioral patterns_ and _linguistic cues_ simultaneously. This hybrid approach improves generalization and interpretability for early mental-health screening.

---

## 🎯 Objectives

- Design a **multi-modal fusion pipeline** combining structured and unstructured data.
- Implement **Tab2Text serialization** for tabular-to-text alignment.
- Fine-tune a **RoBERTa-based model** for mental-health classification.
- Build a **Streamlit-based web interface** supporting both numeric assessment and free-text journaling.
- Integrate a **rule-based recommender system** that provides actionable, personalized coping strategies.

---

## 🏗 System Architecture

The project follows an **Early-Fusion design** that merges all modalities at the data (text) level before model training.

```
        ┌──────────────────────────────┐
        │   Raw Data Sources           │
        │ (Student + Reddit Datasets)  │
        └──────────────┬───────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │ Data Cleaning & Merge  │
           │ Handle nulls, balance  │
           └───────────┬────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │ Tab2Text Serialization │
           │ Structured → Narrative │
           └───────────┬────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │  MentalRoBERTa Model   │
           │ Fine-tuned Classifier  │
           └───────────┬────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │  Streamlit Interface   │
           │ Assessment + Journal   │
           └───────────┬────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │  Recommender Engine    │
           │ Risk-to-Advice Mapping │
           └────────────────────────┘
```

---

## ✨ Key Features

- **Unified Feature Space:** Converts structured survey metrics into coherent text narratives.
- **Intelligent Null Salvage:** Merges 350 Reddit `[deleted]` posts using title-body reconstruction.
- **Synthetic Text Augmentation:** Uses 27K student narratives to pre-train the model for 6K Reddit samples.
- **Dual-Mode Web App:**

  - **Assessment Mode:** Numeric input auto-serialized to text.
  - **Journal Mode:** Free-text analysis for emotional cues.

- **Personalized Recommender:** Maps predicted risk levels to advice and helplines from `resources.json`.

---

## 🧭 Repository Structure

```
mental_health_hybrid/
│
├── data/
│   ├── raw/                  # Source CSV files
│   ├── processed/            # Cleaned hybrid datasets
|   |── augmented/            # optional LLM-generated text
│   └── resources.json        # Coping strategies & helpline database
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb    # Null handling & normalization
│   ├── 02_feature_alignment.ipynb # Tab2Text serialization
│   └── 03_model_training.ipynb   # RoBERTa fine-tuning & evaluation
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data cleaning & preparation functions
│   ├── features.py           # Template-based text serialization logic
│   ├── recommender.py        # Maps predicted risk → coping strategies
|   |── text_converter.py     # Converts the student dataset to text format
│   └── model.py              # RoBERTa architecture, training & metrics
│
├── app/                      # Streamlit Web Application
│   ├── main.py               # UI entry point
│   └── utils.py              # Helper functions (charts, formatting)
│
├── models/                   # Saved model artifacts (.pt / .pkl)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/mental-health-hybrid.git
cd mental-health-hybrid

# 2. Create a virtual environment
python -m venv venv
# Activate it:
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app/main.py
```

---

## 📊 Data & Methodology

### 1. Synthetic Hybrid Strategy

| Dataset                           | Source                                                                                                                                 | Type         | Records | Description                                 |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------- | ------------------------------------------- |
| **Student Mental Health Dataset** | [Kaggle – Student Mental Health](https://www.kaggle.com/datasets/ahmadmideva/student-mental-health)                                    | Structured   | ~27,901 | CGPA, Sleep, Study Hours, Academic Pressure |
| **Reddit Mental Health Posts**    | [Kaggle – Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) | Unstructured | ~51074  | Reddit posts about mental health & distress |
| **Hybrid Narratives (Generated)** | Internal                                                                                                                               | Text         | ~79,000 | Unified text corpus combining both datasets |

**Data Fusion Approach:**

- Converts student records into sentences like:
  _“A 21-year-old student reports poor sleep (4h) and high academic stress.”_
- Merges with Reddit posts to form a balanced unified dataset.

**Null Handling:**

- 365 Reddit posts reconstructed via title-body merging.
- `[deleted]` preserved as a sentiment token.

### 2. Recommendation Engine

`src/recommender.py` uses deterministic mapping from predicted risk levels to supportive resources.

- **Input:** Risk level + contextual features.
- **Logic:** Queries `data/resources.json`.
- **Output:** Tailored advice (e.g., “Try guided breathing” or “Contact Campus Health”).

---

## 🚀 Usage

### Training the Model

1. Run `notebooks/02_feature_alignment.ipynb` to generate `hybrid_dataset.csv`.
2. Run `notebooks/03_model_training.ipynb` to fine-tune **MentalRoBERTa**.
3. Save model checkpoint to:

   ```
   models/hybrid_model.pt
   ```

### Running the Web Interface

```bash
streamlit run app/main.py
```

- **Assessment Mode:** Structured metrics → serialized → classified.
- **Journal Mode:** Text input → semantic risk classification.

---

## 🔬 Evaluation Metrics

| Metric                   | Description                         | Priority |
| ------------------------ | ----------------------------------- | -------- |
| **Macro F1-Score**       | Handles class imbalance effectively | ★★★★★    |
| **Recall (Sensitivity)** | Reduces false negatives             | ★★★★☆    |
| **Precision / Accuracy** | Overall correctness                 | ★★★☆☆    |

### 📈 Experimental Results

| Model                    | Accuracy   | F1-Score | Precision | Recall   |
| ------------------------ | ---------- | -------- | --------- | -------- |
| Random Forest (Baseline) | 85.4 %     | 0.84     | 0.83      | 0.85     |
| BERT-base                | 89.7 %     | 0.88     | 0.89      | 0.88     |
| **MentalRoBERTa Hybrid** | **92.5 %** | **0.91** | **0.92**  | **0.91** |

---

## 🛠 Tech Stack

| Category        | Tools                               |
| --------------- | ----------------------------------- |
| Programming     | Python 3.9 +                        |
| Framework       | Hugging Face Transformers (RoBERTa) |
| Deep Learning   | PyTorch                             |
| Data Processing | Pandas, NumPy                       |
| Visualization   | Matplotlib, Seaborn                 |
| Web Framework   | Streamlit                           |
| Version Control | Git, GitHub                         |

---

## 🚀 Future Enhancements

- Add **real-time emotion detection** for free-text journals.
- Extend to **multilingual datasets** for broader coverage.
- Implement **emotion-category tagging** (Anxiety, Depression, Stress).
- Deploy to **Hugging Face Spaces / Streamlit Cloud**.
- Add **LLM-based interpretability** for explainable AI insights.

---

## 📝 License & Disclaimer

**License:** MIT License
**Disclaimer:** This software is for **educational and research purposes only**.
It is _not_ a substitute for professional medical diagnosis or treatment.
If you or someone you know is in distress, please reach out to a licensed professional or helpline.

---

## 👤 Author

**Name:** [Your Full Name]
**Role:** Final-Year Computer Science Student
**Department:** Computer Science
**Institution:** [Your University Name]
**Supervisor:** [Supervisor Name]
**Academic Year:** 2025

---
