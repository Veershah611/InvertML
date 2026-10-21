# Section 15: AI-Usage, External Resources & Academic Integrity Log

**Document Owner:** Partner B2 (Evaluation Architect & Documentation Co-Lead)  
**Contributing Authors:** All Team Members (Pair A: A1, A2 · Pair B: B1, B2)  
**Assignment Reference:** Innovative Assignment Guidelines §3, §10, §15 & [`Solar_Inverter_Health_Monitoring.md`](../Solar_Inverter_Health_Monitoring.md)

---

## 15.1 Academic Integrity Statement

The team explicitly certifies the following principles of academic integrity:
1. **Zero Fabrication:** Every metric, table value, confusion matrix, and curve presented in this project was produced by executing reproducible Python code on the project datasets. No numbers, simulation outputs, or validation metrics were hallucinated, estimated, or fabricated.
2. **Deterministic Reproducibility:** Fixed random seeds (`random_state=42`) were enforced across all data splitters, transformers, cross-validation folds, and model estimators.
3. **Traceability:** Every row in `results/results.csv` corresponds to a timestamped model execution using the unified `src/evaluate.py` logging engine.

---

## 15.2 Generative AI Usage Log

In accordance with institutional guidelines on transparent artificial intelligence usage, the following log details all generative AI interactions during the project lifecycle:

| Date | Member | AI Tool / Model | Task / Prompt Category | How the Output Was Verified |
|---|---|---|---|---|
| Oct 18, 2026 | A1 | Gemini 3.8 / Claude | SPARQL query template for querying Brick schema Turtle (`.ttl`) metadata | Query tested in `src/prepare_data.py`; verified that returned capacity and inverter counts matched physical campus inventory. |
| Oct 19, 2026 | B2 | Gemini 3.8 Flash | Scaffolding standardized evaluation metrics functions (`evaluate.py`) | Executed `python -m src.evaluate`; confirmed mathematical parity with standard `sklearn.metrics` formulas on synthetic dummy arrays. |
| Oct 19, 2026 | B2 | Gemini 3.8 Flash | Matplotlib plotting boilerplate for seasonal decomposition and boxplots | Inspected rendered `.png` outputs in `reports/figures/`; confirmed axes labels, scale limits, and statistical percentiles. |
| Oct 20, 2026 | B1 | Claude 3.5 Sonnet | LaTeX mathematical formulas for XGBoost second-order Taylor expansion | Cross-referenced against Chen & Guestrin (2016) original XGBoost paper equations; confirmed math correctness. |
| Oct 20, 2026 | A2 | Claude 3.5 Sonnet | Drafting data-leakage audit framework and cross-validation boundary verification | Inspected `src/preprocess.py` pipeline to ensure no test-fold statistics entered scalers or imputers. |
| Oct 21, 2026 | B2 | Gemini 3.8 Flash | Markdown table formatting and Stage 12 impact template | Reviewed and customized all operational stakeholder details against the HKUST solar plant context. |

### Strict Rules Followed for AI Assistance:
* **No Code Black-Boxes:** Any AI-assisted code snippet was peer-reviewed by the respective partner and executed locally with unit tests before pushing to GitHub.
* **No Synthetic Metrics:** AI was strictly prohibited from generating performance numbers or test scores; all results in `results/results.csv` originate directly from Python execution logs.

---

## 15.3 External Datasets & Licensing

| Dataset Name | Source / Author | DOI / Link | License | Role in Project |
|---|---|---|---|---|
| **HKUST Rooftop PV Dataset (2021–2023)** | Lin et al., *Scientific Data* (2025) | [10.5061/dryad.m37pvmd99](https://doi.org/10.5061/dryad.m37pvmd99) | **CC0 (Public Domain)** | Primary training, validation, and testing dataset (44 inverters, 3 years). |
| **GPVS-Faults Dataset** | Bakdi et al., *Mendeley Data* (2021) | [10.17632/n76t439f65.1](https://doi.org/10.17632/n76t439f65.1) | **CC BY 4.0** | Secondary benchmark dataset for physical fault validation (F0–F7 microgrid faults). |

---

## 15.4 Pinned Software Dependencies

All experiments were executed on CPU laptops running Python 3.12/3.14 with the pinned environment defined in [`requirements.txt`](../requirements.txt):

* `pandas==3.0.6`: Dataframe operations, resampling, and time indexing.
* `numpy==2.5.4`: Numerical linear algebra and vectorized array transformations.
* `pyarrow==26.0.0`: High-speed column-oriented Parquet file storage.
* `scikit-learn==1.9.1`: Preprocessing pipelines, Ridge, Logistic Regression, Random Forest, MLP, and Isolation Forest.
* `xgboost==3.1.1`: Extreme Gradient Boosting tree models with histogram optimization (`tree_method="hist"`).
* `rdflib==7.6.0`: Brick Schema RDF graph parsing and SPARQL query execution.
* `statsmodels==0.15.0`: Time series seasonal decomposition.
* `matplotlib==3.11.2` & `seaborn==0.13.2`: Visualizations and publication-grade figures.
* `openpyxl==3.1.5`: Reading historical rainfall spreadsheet files.
* `PyYAML==6.0.3`: Central configuration management (`configs/config.yaml`).
