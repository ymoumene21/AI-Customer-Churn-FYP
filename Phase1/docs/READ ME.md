# 🧠 AI-Enabled Customer Churn Prediction Platform
**Liverpool John Moores University – Final Year Project (2025–2026)**  
Author: Mr. Stark

---

## 🎯 Project Overview
This project builds an **AI-driven churn prediction system** using the **IBM Telco Customer Churn Dataset**.  
It applies **machine learning and explainable AI techniques** to identify customers likely to leave a telecom service and provides actionable insights to support retention decisions.

---

## ⚙️ Key Objectives
- ✅ Define project requirements and scope (Phase 1)
- ✅ Conduct literature review on telecom churn prediction (2018–2025)
- ✅ Analyze dataset and feature importance (IBM Telco)
- 🔄 Develop predictive models (Phase 2 – in progress)
- 🧠 Implement explainability (SHAP) and deploy dashboard/API (Phase 4)

---

## 🧩 Project Phases
| Phase | Description | Status |
|:--|:--|:--:|
| **Phase 1** | Requirement Analysis & Literature Review | ✅ Complete |
| **Phase 2** | Data Preprocessing & Model Development | 🔄 In progress |
| **Phase 3** | Evaluation & Explainability | ⏳ Upcoming |
| **Phase 4** | Dashboard + API Integration | ⏳ Upcoming |

---

## 📊 Dataset (IBM Telco)
- **Rows:** 7,043  **Columns:** 21  
- **Target:** `Churn` (Yes = 1, No = 0)  
- **Churn Rate:** ≈ 26.5%  
- **File:** `Phase1/data/IBM_Telco_Customer_Churn_raw.csv` (not tracked in Git)

---

## 📈 Evaluation Metrics & Acceptance (locked in Phase 1)
| Metric | Purpose | Threshold |
|:--|:--|:--:|
| **ROC–AUC** | Model discrimination power | **≥ 0.83** |
| **F1 Score** | Balance Precision & Recall on churners | **≥ 0.60** |
| Accuracy | Overall correctness (sanity) | Secondary |
| SHAP | Explainability (global + local) | Qualitative |

**Reporting protocol:** stratified split, 5-fold CV, untouched hold-out test; confusion matrix, ROC/PR curves; threshold at max F1.

---

## 📚 Literature Insights (2018–2025)
- **Contract Type, Tenure, MonthlyCharges** are the most consistent churn drivers.  
- **Ensembles (XGBoost/Random Forest/LightGBM)** dominate; **DL hybrids (CNN/RNN)** show modest gains with higher complexity.  
- **Explainable AI (SHAP/LIME)** is increasingly required for adoption.


---

## 📅 Phase 1 Progress Log (Requirement Analysis & Literature Review)

| **Day** | **Focus Area** | **Key Activities** | **Deliverables / Outputs** |
|:--:|:--|:--|:--|
| 🧠 **Day 1** | **Environment Setup & Dataset Validation** | • Created project structure (`Phase1/data`, `docs`, `notes`, `papers`, `code`) • Set up virtual environment • Imported & validated IBM Telco dataset (7,043×21) • Checked churn distribution (Yes ≈ 26.5%) | ✅ `code/fyp.py` • ✅ Notion Data Map • ✅ Initial GitHub commit |
| 📚 **Day 2** | **Literature Review Setup** | • Collected 8 papers (5 General Telecom + 3 IBM Telco) • Standardized filenames & folders • Built summary table | ✅ `Phase1/papers/` organized • ✅ `Phase1/docs/Day2_Literature_Review_Setup.docx` |
| 📊 **Day 3** | **Metrics & Acceptance + Traceability** | • Locked metrics (Accuracy/Precision/Recall/F1/ROC–AUC) • Set thresholds (AUC ≥ 0.83, F1 ≥ 0.60) • Built RTM • Drafted Risk Register & Test Plan | ✅ `Phase1/docs/Day3_Metrics_Traceability.md` |
| 🧮 **Day 4** | **Literature Deep Dive & Feature Analysis** | • Compared research features vs IBM Telco • Generated dataset stats & correlations • Drafted Phase 2 Feature Plan | ✅ `reports/day4_summary.csv` • ✅ `reports/day4_missing.csv` • ✅ `reports/day4_corr.csv` • ✅ `Phase1/docs/Day4_Feature_Analysis.docx` |

### 🧠 End of Phase 1 Summary
- Requirements fully defined (FR & NFR).  
- Dataset validated and understood.  
- Metrics + acceptance thresholds locked.  
- Literature mapped to dataset features.  
- ✅ Ready to begin **Phase 2: Data Preprocessing & Model Development**.

---

## 🧠 Technologies
- Python 3.12, pandas, scikit-learn, xgboost, matplotlib  
- SHAP (explainability)  
- Streamlit / FastAPI (planned deployment)  
- Git + GitHub (version control)

---

## 📂 Repository Structure
