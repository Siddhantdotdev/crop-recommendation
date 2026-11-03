# 🌱 AI-Based Crop Recommendation System Using Soil Analysis Parameters

An end-to-end Machine Learning college mini-project that predicts the optimal crop to cultivate based on soil chemical parameters ($N$, $P$, $K$, $pH$) and environmental/weather conditions (Temperature, Humidity, Rainfall).

---

## 📌 1. Project Objective & Motivation

Agriculture is the backbone of food security and rural livelihoods. However, traditional farming frequently relies on manual guesswork, inherited habits, or trial-and-error, leading to:
- Mismatch between soil nutrient profile and crop physiological requirements.
- Depletion of vital soil macronutrients ($N$, $P$, $K$).
- Poor crop yields and severe economic loss for farmers.

**Objective:**  
To develop a high-accuracy, reliable **Machine Learning Decision Support System** that analyzes laboratory soil tests and regional weather forecasts to recommend the highest-yielding crop with accompanying confidence scores.

---

## 📊 2. Dataset Description

The system uses the Kaggle **Crop Recommendation Dataset** (`Crop_recommendation.csv`), comprising **2,200 soil and atmospheric observations** across **22 distinct crops**.

### Features (Input Variables $X$):
| Feature | Parameter | Unit | Description | Min - Max Range |
|---|---|---|---|---|
| `N` | Nitrogen | kg/ha | Essential macronutrient for plant leaf growth & chlorophyll. | 0 - 140 |
| `P` | Phosphorus | kg/ha | Key nutrient for root development, seed, and flower formation. | 5 - 145 |
| `K` | Potassium | kg/ha | Regulates water retention, drought tolerance, and disease immunity. | 5 - 205 |
| `temperature` | Air Temperature | °C | Ambient atmospheric temperature required for germination. | 8.83 - 43.68 |
| `humidity` | Relative Humidity | % | Water vapor ratio in ambient air. | 14.26 - 99.98 |
| `ph` | Soil pH | Scale | Acidity/alkalinity level of soil (Neutral: 6.0 - 7.5). | 3.50 - 9.94 |
| `rainfall` | Precipitation | mm | Average seasonal water precipitation. | 20.21 - 298.56 |

### Target Label (Output Variable $y$):
- **22 Unique Crops**: `rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`.
- **Class Balance**: Exactly **100 records per crop** (balanced multiclass classification problem).
- **Missing Values**: 0
- **Duplicate Rows**: 0

---

## 🧠 3. Machine Learning Algorithms Evaluated

To find the optimal classifier, three distinct algorithmic paradigms were benchmarked:

1. **Decision Tree Classifier (`DecisionTreeClassifier(random_state=42)`)**:
   - Non-parametric recursive splitting based on Gini Impurity.
   - Fast and transparent, but prone to high variance on complex boundary conditions.

2. **Random Forest Classifier (`RandomForestClassifier(n_estimators=100, random_state=42)`)**:
   - Ensemble bagging method constructing 100 decorrelated decision trees.
   - Aggregates probability outputs through majority voting, effectively eliminating overfitting.

3. **K-Nearest Neighbors (`KNeighborsClassifier(n_neighbors=5)`)**:
   - Instance-based distance metric algorithm.
   - **Pipeline Integration**: Implemented via an `sklearn.pipeline.Pipeline` with `StandardScaler` to normalize feature scales (e.g. rainfall ~300 mm vs pH ~7) and strictly prevent data leakage.

---

## ⚙️ 4. Data Preprocessing & Methodology

```
+---------------------------+
|  Crop_recommendation.csv  |
+---------------------------+
              |
              v
+---------------------------+
|  Data Exploration & Audit | (0 nulls, 0 duplicates, 22 balanced classes)
+---------------------------+
              |
              v
+---------------------------+
|  80:20 Stratified Split   | (random_state=42, stratify=y)
|  Train: 1760 | Test: 440  | (Exact 80 train / 20 test per crop)
+---------------------------+
              |
              v
+---------------------------+
|   Leakage-Free Pipelines  | (StandardScaler fitted strictly on X_train)
+---------------------------+
              |
              v
+---------------------------+
|  Train 3 ML Classifiers   | (Decision Tree, Random Forest, Scaled KNN)
+---------------------------+
              |
              v
+---------------------------+
|  Model Evaluation on Test | (Accuracy, Weighted Precision, Recall, F1)
+---------------------------+
              |
              v
+---------------------------+
| Model Persistence (Joblib)| -> crop_model.pkl (Random Forest: 99.55%)
+---------------------------+
```

### Prevention of Data Leakage:
- Any transformation (such as `StandardScaler` for KNN) calculates the mean ($\mu$) and standard deviation ($\sigma$) strictly on the training set (`X_train`).
- The test set (`X_test`) is treated as completely unseen data and only transformed using the training parameters.

---

## 🏆 5. Real Evaluation Results on Test Set (440 Samples)

The models were evaluated on the **20% unseen test holdout split (440 samples)**. All metrics below are genuine, verified figures computed directly on the dataset:

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Misclassified Samples |
|---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Winner)** | **99.55%** | **99.57%** | **99.55%** | **99.55%** | **2 / 440** |
| **Decision Tree** | 97.95% | 98.06% | 97.95% | 97.94% | 9 / 440 |
| **K-Nearest Neighbors (Scaled)** | 97.95% | 98.04% | 97.95% | 97.93% | 9 / 440 |

### Why Random Forest Won:
- Out of 440 unseen test instances, Random Forest correctly classified **438 samples (99.55%)**.
- Its ensemble of 100 randomized trees smooths out boundary noise between closely related agro-climatic crops (e.g., distinguishing between rice and jute in humid conditions).
- Consequently, `RandomForestClassifier` was selected and serialized as `crop_model.pkl`.

---

## 🚀 6. How to Execute the Project

### Prerequisites:
Ensure Python 3.10+ (or Python 3.14) is installed.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train & Evaluate Models
Run the training script to load data, perform exploratory analysis, benchmark all 3 models, and export `crop_model.pkl`:
```bash
python train.py
```

### Step 3: Run Command-Line Inference (`predict.py`)

**Option A — Run built-in demonstration test cases:**
```bash
python predict.py
```

**Option B — Run with custom soil & weather parameters:**
```bash
python predict.py --N 90 --P 42 --K 43 --temperature 20.87 --humidity 82.0 --ph 6.5 --rainfall 202.9
```
*Output:*
```text
============================================================
Recommended Crop : RICE
Top 3 Recommendations:
  1. rice         : 90.00%
  2. jute         : 10.00%
  3. pomegranate  : 0.00%
============================================================
```

### Step 4: Launch Interactive Streamlit Web App (`app.py`)
Launch the web interface for demonstration to examiners:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.  
Features:
- Soil & environmental sliders with dataset range limits.
- One-click demonstration presets (Rice, Apple, Coffee).
- Visual confidence cards and Top-3 candidate comparison bar chart.
- Model performance metrics and architectural summary.

### Step 5: View or Run Jupyter Notebook
Launch the comprehensive notebook containing all exploratory charts, heatmaps, and metrics:
```bash
jupyter lab Crop_Recommendation.ipynb
# or
jupyter notebook Crop_Recommendation.ipynb
```

---

## 📁 7. Project File Structure

```text
crop-recommendation/
│
├── Crop_recommendation.csv     # Kaggle dataset (2200 rows, 8 columns)
├── Crop_Recommendation.ipynb   # Complete, fully executed Jupyter Notebook with plots
├── train.py                    # Model training, evaluation, and serialization script
├── predict.py                  # CLI inference script returning top-3 crop probabilities
├── app.py                      # Interactive Streamlit web application
├── crop_model.pkl              # Serialized Random Forest model (Joblib)
├── requirements.txt            # Python package dependencies
├── viva_questions.md           # 20+ viva preparation Q&A for external examination
└── README.md                   # Complete project documentation
```

---

## 👨‍💻 Author & Examination Note
This project was prepared for the **B.Tech / B.E. Computer Science & Engineering Practical Examination**. All machine learning pipelines follow strict Scikit-learn best practices, zero data leakage, and verified reproducible metrics.

