# B2 Work Guide: Solar Inverter Health Monitoring System

---

## 1. Overview of B2's Role

In this 4-person team project, the team is divided into two pairs:
* **Pair A (A1, A2):** Data Foundation (Raw data extraction, cleaning, preprocessing, balancing).
* **Pair B (B1, B2):** Features & Evaluation (Feature design, target formulation, validation engine).

### What is B2's Core Identity?
**B2 is the "Evaluation Architect & Neural Network Specialist".**
B2 has two major responsibilities:
1. **The Shared Evaluation Framework (Stage 11 & 12 Lead):** You build the central rules and code (`evaluate.py`) that everyone uses to test their models fairly. You also design how we test on completely new (unseen) solar stations and analyze real-world impact and limitations.
2. **The Neural Network Modeling Track (Track T4):** You own the Multi-Layer Perceptron (MLP) model, testing if deep learning / neural network approaches beat traditional tree models (Random Forest, XGBoost) and linear baselines.

---

## 2. Complete Summary of Assigned Tasks for B2

| Area | Specific Task | Lead or Shared? | Key Output / Code File |
|---|---|---|---|
| **Pipeline Lead** | Stage 11: Validation & Evaluation Framework | **B2 Leads** (Used by all 4) | `src/evaluate.py`, `results/results.csv` |
| **Pipeline Lead** | Stage 12: Use Case, Impact & Limitations | **B2 Leads** | Final Report Sections 11 & 14 |
| **EDA (§6)** | 1. Analyze 3-year Seasonality<br>2. Analyze Class Distribution | B2 individual EDA | `notebooks/01_eda_B2.ipynb` |
| **Feature Engineering (§7)** | Build Time Features (Sin/Cos) & Station Metadata features | B2 individual contribution | Added to `src/features.py` |
| **Supervised Modeling (§9)** | Multi-Layer Perceptron (MLP) for **Regression** and **Classification** | B2 owns Track T4 | `src/models/mlp.py`, `notebooks/04_mlp_B2.ipynb` |
| **Hyperparameter Tuning (§10)**| Tune MLP parameters using `TimeSeriesSplit` on a sample | B2 owns MLP tuning | Tuning section in notebook & results table |
| **Model Explainability (§11.3)** | Permutation Importance & Partial Dependence Plots (PDP) | B2 owns for MLP | Plots & failure case analysis in report |
| **Anomaly Detection (§9/11)**| Isolation Forest tested on **GPVS-Faults** (Real Fault Dataset) | Shared with partner B1 | GPVS loader + Isolation Forest evaluation |
| **Documentation & Ethics (§15)**| AI-Usage & External Resources Log | B2 writes | Final report integrity section |

---

## 3. Deep Dive into Each Task: Algorithms & Step-by-Step Execution

---

### Task 1: Exploratory Data Analysis (EDA) — 2 Questions

You must analyze the data to answer two specific operational questions and turn those insights into engineering decisions.

#### Question 1: How strong is seasonality across the 3 years (2021–2023)?
* **Goal in simple words:** Solar energy obviously changes between summer and winter. We need to measure how much power generation varies across months so our models don't confuse normal winter low light with a broken inverter.
* **Algorithm / Technique:** 
  * Group data by station and month.
  * Compute **Specific Yield** = $\frac{\text{Total AC Energy (kWh)}}{\text{Rated Capacity (kWp)}}$.
  * Plot monthly box plots and time-series line charts across all 3 years.
* **Decision it informs:**
  * Proves why a standard random train/test split would cause data leakage.
  * Justifies the chronological split (putting all 4 seasons of 2023 in the final test set).
  * Justifies adding cyclic seasonal features (day of year).

#### Question 2: How are the operational classes distributed across seasons and stations?
* **Goal in simple words:** Check how many data points belong to `Healthy`, `Warning`, and `Faulty`. Is `Faulty` super rare? Do certain stations fail way more often?
* **Algorithm / Technique:**
  * Value counts and percentage breakdowns of the engineered health labels (`Healthy`, `Warning`, `Faulty`).
  * Cross-tabulation / stacked bar charts: Class frequency per station and class frequency per season.
* **Decision it informs:**
  * Confirms the level of class imbalance (e.g., `Faulty` might be < 5% of the dataset).
  * Informs Pair A on data balancing choices and tells you why **Accuracy is a misleading metric** (you will focus on **Recall for Faulty**).

---

### Task 2: Feature Engineering — Time & Station Metadata

You will write the code for your feature group and plug it into the shared `src/features.py`.

#### 1. Cyclical Time Features (Hour and Day of Year)
* **The Problem:** A standard number for hour (0 to 23) tricks a model into thinking hour 23 (11:00 PM) and hour 0 (midnight) are as far apart as possible, even though they are just 1 hour apart.
* **The Solution (Algorithm):** Sine and Cosine cyclic encoding.
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \times \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \times \text{hour}}{24}\right)$$
  $$\text{day\_sin} = \sin\left(\frac{2\pi \times \text{day\_of\_year}}{365.25}\right), \quad \text{day\_cos} = \cos\left(\frac{2\pi \times \text{day\_of\_year}}{365.25}\right)$$
* **Why it matters:** Neural networks and linear models understand cyclic circular patterns instantly with this conversion.

#### 2. Station Metadata Features
* **What you use:** Station capacity (kWp), roof orientation (azimuth angle), and tilt angle (extracted from the Brick metadata parsed by A1).
* **How it is performed:**
  * Normalize numerical capacity.
  * One-hot encode or categorize categorical orientation (e.g., South-facing vs East/West-facing).
* **Why it matters:** Allows the model to understand why a 10 kW station generates less raw power than a 50 kW station without mistaking it for degradation.

---

### Task 3: Supervised Modeling — Multi-Layer Perceptron (MLP)

You own the **Neural Network paradigm (Track T4)**. You will build and test two models:
1. **MLP Regressor:** Predicts a continuous Health Score from `0 to 100`.
2. **MLP Classifier:** Predicts the status class (`Healthy`, `Warning`, `Faulty`).

#### Algorithm Details: Multi-Layer Perceptron
* **Library:** `sklearn.neural_network.MLPRegressor` and `sklearn.neural_network.MLPClassifier`.
* **Architecture:** Feedforward Artificial Neural Network with:
  * Input layer: ~25–30 engineered features.
  * Hidden layers: 2 hidden layers (e.g., `(64, 32)` or `(100, 50)` neurons).
  * Activation function: `relu` (Rectified Linear Unit) for hidden neurons.
  * Output layer:
    * For Regression: 1 linear output unit (Health Score).
    * For Classification: Softmax output across 3 classes (`Healthy`, `Warning`, `Faulty`).
  * Optimizer / Solver: `adam` (Adaptive Moment Estimation).
  * Loss Function:
    * Regression: Mean Squared Error (`squared_error`).
    * Classification: Cross-Entropy / Log-loss (`log_loss`).
  * Regularization: L2 regularization penalty (`alpha`), early stopping with validation fraction.

#### Crucial Step: Mandatory Scaling
* **Why:** Unlike tree algorithms (Random Forest, XGBoost), **neural networks will fail or train extremely slowly if inputs are not scaled**.
* **How:** Wrap the model inside an `sklearn.pipeline.Pipeline`:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.neural_network import MLPRegressor

  mlp_pipeline = Pipeline([
      ('scaler', StandardScaler()),
      ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42))
  ])
  ```
* **Leakage Rule:** The `StandardScaler` must fit **only on training data** and merely transform validation/testing data.

#### Bonus Architectural Comparison:
Compare two ways to classify status:
* **Method 1:** Train a dedicated `MLPClassifier`.
* **Method 2:** Use the `MLPRegressor`'s predicted Health Score and apply thresholds (e.g., $\ge 90 \to \text{Healthy}$, $70-90 \to \text{Warning}$, $< 70 \to \text{Faulty}$).
* Report which approach delivers higher `Faulty` recall!

---

### Task 4: Hyperparameter Tuning for MLP

Because neural networks can take hours to train on full datasets, you must follow the smart compute plan.

* **Sampling Strategy:** Extract a **10% to 20% time-ordered sample** from the training period (~200k to 500k rows) so your laptop doesn't overheat.
* **Cross-Validation Algorithm:** `TimeSeriesSplit(n_splits=3 or 5)` (Expanding Window).
  * *Never use standard K-Fold shuffle*, as that leaks future data into past predictions!
* **Search Strategy:** `RandomizedSearchCV` (explores 15–20 parameter combinations).
* **Parameters to Search:**
  * `hidden_layer_sizes`: `[(64, 32)`, `(100, 50)`, `(32, 16)`, `(128, 64, 32)]`
  * `alpha` (L2 penalty): `[1e-4, 1e-3, 1e-2, 0.1]`
  * `learning_rate_init`: `[0.001, 0.005, 0.01]`
  * `batch_size`: `[256, 512, 1024]`
* **Deliverable:** A clear comparison table showing **Default vs. Tuned** performance on the validation set. Once best parameters are found, fit the final model on the full training set.

---

### Task 5: Unsupervised Anomaly Detection on Real Faults (with B1)

While Pair A tests Isolation Forest on HKUST with synthetic injected faults, **Pair B tests Isolation Forest on the GPVS-Faults dataset**.

* **Why this is critical:** The HKUST labels are mathematically engineered. GPVS-Faults contains **real physical faults** recorded in a PV microgrid lab experiment (shading, open circuits, short circuits, inverter disconnection). This proves whether our anomaly detection really works in real life!
* **Algorithm:** `sklearn.ensemble.IsolationForest`.
  * *How it works in simple words:* It builds random decision trees. Anomalies (faults) are rare and have unusual current/voltage patterns, so they get isolated in very few splits (near the root of the tree). Normal points require many splits to isolate.
* **How you will perform it:**
  1. Write a loader script for GPVS-Faults `.csv` / `.mat` files (`F0` = normal, `F1`–`F7` = faults).
  2. Compute simple window statistics (mean, std of DC voltage, AC power, currents).
  3. Fit `IsolationForest` on normal data (`F0`) or mixed data.
  4. Predict anomalies (-1 = anomaly, 1 = normal).
  5. Calculate: **Precision, Recall, and F1-Score** against the true fault labels.

---

### Task 6: Master Validation & Evaluation Framework (`evaluate.py`)

This is B2's main leadership role. You write the shared evaluation module that all 4 teammates import.

#### 1. Chronological Split Setup
Ensure no data from the future is ever used to predict the past:
* **Training Period:** January 1, 2021 – June 30, 2022 (18 months).
* **Validation Period:** July 1, 2022 – December 31, 2022 (6 months, for model tuning).
* **Test Period:** January 1, 2023 – December 31, 2023 (12 months, full year containing all seasons).

#### 2. The "Unseen Station" Generalization Test
* **What it is:** Select ~7 rooftop stations out of 37 (~20%) and **completely remove them from the training set**.
* *Natural Candidates:* The dataset notes that 6 inverters only started recording in 2023 (e.g., SQ Block P, Shaw Auditorium, UG Hall 8 & 9).
* **How it works:** Models train only on the remaining 30 stations. Then, test models on these 7 unseen stations in 2023.
* **Why it matters:** Answers the viva question: *"If our university installs solar panels on a brand-new building tomorrow, will your ML model work without retraining?"*

#### 3. Standardized Metrics Function in `evaluate.py`
You will build functions that output:
* **For Regression:**
  * **MAE (Mean Absolute Error):** How many points off the health score is on average (e.g., $\pm 3.2$ points).
  * **RMSE (Root Mean Squared Error):** Penalizes rare, catastrophic prediction errors.
  * **$R^2$ Score:** How much variance in health score is explained.
* **For Classification:**
  * **Macro-F1 Score:** Treats all classes equally, preventing the dominant `Healthy` class from hiding bad performance.
  * **Recall for `Faulty` (The Most Important Metric):** If an inverter is dying, did we catch it? Missing a broken inverter is far more costly than a false alarm.
  * **PR-AUC (Precision-Recall Area Under Curve):** Crucial for severely imbalanced data.
  * **Confusion Matrix:** True vs. Predicted grid.

#### 4. Shared Results Logger
Create a simple helper in `evaluate.py` that writes a row into `results/results.csv` after every model evaluation:
```
timestamp, student, model_name, task, train_score, val_score, test_score, unseen_station_score
```
This ensures your team has an automated, identical comparison table ready for the final report.

---

### Task 7: Model Explainability & Error Analysis

Evaluators want to see *why* your neural network makes decisions, not just black-box numbers.

* **Technique 1: Permutation Importance**
  * *How it works in simple words:* Take a feature (e.g., `peer_deviation`), shuffle its values randomly so it becomes garbage, and measure how much the model's accuracy drops. If accuracy crashes, that feature was critical!
* **Technique 2: Partial Dependence Plots (PDP)**
  * *How it works:* Shows the marginal effect that 1 or 2 features have on the predicted Health Score (e.g., as ambient temperature rises above 35°C, how does predicted health curve downward?).
* **Failure Case Analysis (Qualitative Error Review):**
  * Find the top 3 worst predictions made by your MLP.
  * Investigate: Was it a cloudy day? Was the station undergoing maintenance? Was there a data gap? Explain these in the report.

---

### Task 8: Stage 12 — Use Case, Impact, Limitations & AI Log

You lead the narrative write-up for the real-world deployment chapter.

1. **Operational Impact:**
   * Explain how campus facility teams use the 15-minute dashboard to triage maintenance tickets rather than inspecting 37 rooftops manually.
2. **Honest Limitations:**
   * Only one weather station for 37 distinct roofs (local micro-shading from trees/elevators isn't detected by the weather sensor).
   * Labels on HKUST are rule-engineered, which is why GPVS-Faults was necessary to validate physical faults.
3. **AI-Usage & Resources Log (§15):**
   * Keep a transparent log of prompts and tools used (e.g., Claude/ChatGPT/Gemini for boilerplate debugging, libraries used, and how outputs were verified).

---

## 4. B2 Deliverables & Code File Checklist

```
solar-health/
├── src/
│   ├── evaluate.py         <-- B2 CREATES & OWNS (Master evaluation & splitting logic)
│   ├── features.py         <-- B2 ADDS (Time sin/cos & metadata features)
│   ├── models/
│   │   ├── mlp.py          <-- B2 CREATES & OWNS (MLPRegressor & MLPClassifier pipelines)
│   │   └── iforest.py      <-- B2 CO-OWNS with B1 (Evaluates on GPVS-Faults)
├── notebooks/
│   ├── 01_eda_B2.ipynb     <-- B2 CREATES (Seasonality & class distribution plots)
│   └── 04_mlp_B2.ipynb     <-- B2 CREATES (Tuning, training, evaluation & PDP plots)
├── results/
│   └── results.csv         <-- B2 sets up header & logs MLP results
└── reports/
    ├── Section_11_Evaluation.md <-- B2 writes
    ├── Section_14_Impact.md     <-- B2 writes
    └── AI_Usage_Log.md          <-- B2 writes
```

---

## 5. Step-by-Step 4-Week Schedule for B2

### Week 1: Evaluation Setup & EDA
* [ ] Implement chronological splitting and the unseen-station holdout list in `src/evaluate.py`.
* [ ] Write standard evaluation functions (MAE, RMSE, $R^2$, Macro-F1, Confusion Matrix).
* [ ] Inspect GPVS-Faults dataset and write basic data loader.
* [ ] Complete EDA Question 1 (Seasonality yield plots) and EDA Question 2 (Class distributions).

### Week 2: Features, Baseline Run & Framework Freeze
* [ ] Write cyclic time features (`hour_sin`, `hour_cos`, `day_sin`, `day_cos`) and metadata encoders in `src/features.py`.
* [ ] Freeze `src/evaluate.py` so all teammates can test their Week 2 models.
* [ ] Build first untuned MLP baseline with `StandardScaler` on a 10% sample; verify pipeline runs without errors.

### Week 3: Tuning, Anomaly Testing & Explainability
* [ ] Run `RandomizedSearchCV` on MLP using `TimeSeriesSplit` on the sample; select best hyperparameters.
* [ ] Refit best MLP on full training dataset; compute test scores on 2023 test set and unseen-station set.
* [ ] Work with B1: Run `IsolationForest` on GPVS-Faults and compute fault detection recall/precision.
* [ ] Generate Permutation Importance rankings and Partial Dependence Plots for the MLP.
* [ ] Identify 3 specific failure cases (worst errors).

### Week 4: Final Reporting & Viva Drill
* [ ] Assemble all team scores into the master comparison table in `results/results.csv`.
* [ ] Draft Section 11 (Validation & Evaluation) and Section 14 (Use Case, Impact & Limitations).
* [ ] Compile the AI-usage and references log.
* [ ] Review B1's XGBoost code and prepare for peer explanation in the oral exam / viva.

---

## 6. Viva / Oral Exam Quick Cheat Sheet for B2

Be ready to answer these questions confidently:

1. **"Why did you use TimeSeriesSplit instead of regular 5-fold cross-validation?"**
   * *Answer:* Solar data is time-dependent. Shuffling data leaks future trends into past predictions. `TimeSeriesSplit` always trains on past data and predicts on subsequent future time slices.
2. **"Why did your MLP model need StandardScaler while your partner's Random Forest didn't?"**
   * *Answer:* Tree-based models only look at split thresholds, so magnitude doesn't matter. Neural networks use gradient descent and weight dot-products; unscaled features with large ranges (like raw Active Power in Watts) dominate the gradients and prevent convergence.
3. **"Why is Faulty Recall your priority metric over Accuracy?"**
   * *Answer:* Faults represent only ~5% of the data. A naive model that always predicts "Healthy" achieves 95% accuracy but is 100% useless because it never detects a broken inverter. Missing a fault causes real revenue loss.
4. **"What does the Unseen-Station Test prove?"**
   * *Answer:* It proves spatial generalization. Rooftops have different tilts, capacities, and shading. By testing on 7 stations never seen during training, we prove our model learns general inverter degradation rather than just memorizing specific rooftops.
