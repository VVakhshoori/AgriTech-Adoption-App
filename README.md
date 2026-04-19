# 🌾 Agri-Tech Policy Adoption Simulator

A **Streamlit-based machine learning application** that simulates the likelihood of adopting agricultural technologies under different policy and production scenarios. The system leverages an **ensemble of Random Forest models** to estimate adoption probability and uncertainty based on farmers’ socioeconomic characteristics.

---

## 🚀 Live App

👉 *Add your deployed link here after publishing on Streamlit Cloud*
(e.g., https://agritech-adoption-app.streamlit.app)

---

## 📊 Project Overview

This tool enables users to:

* Simulate **agri-technology adoption probability**
* Evaluate the impact of:

  * Policy types (Financial, Education, R&D, Market, etc.)
  * Production scenarios (Increase / Decrease)
  * Technology types (Digital, CEA, Genomic)
* Input farmer socioeconomic profiles interactively
* Run **ensemble predictions across multiple trained models**
* Perform **batch analysis** using uploaded datasets
* Visualize prediction results and uncertainty ranges

---

## 🧠 Machine Learning Approach

* Algorithm: **Random Forest (Scikit-learn)**
* Structure: **Ensemble of models per scenario-policy combination**
* Outputs:

  * Mean adoption probability
  * Probability range (min–max across models)
  * Ensemble AUC score

---

## 📁 Repository Structure

```text
AgriTech-Adoption-App/
│
├── app.py                         # Main Streamlit application (entry point)
├── requirements.txt               # Python dependencies
├── Ensemble_Committee_Models/     # Trained ML models
└── README.md
```

---

## ⚙️ Installation (Local Setup)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AgriTech-Adoption-App.git
cd AgriTech-Adoption-App
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## 📦 Requirements

* streamlit
* pandas
* numpy
* scikit-learn
* joblib
* plotly
* xlsxwriter
* openpyxl

---

## 📥 Input Features

The model uses the following socioeconomic variables:

* Age
* Total farm area
* Employee number
* Farm revenue
* Asset value
* Hours on/off farm
* Education level
* Gender
* Immigration status
* Indigenous status

---

## 📊 Outputs

### 🔹 Single Prediction Mode

* Adoption probability
* Adoption category (Low / Moderate / Strong)
* Probability range across ensemble models
* Ensemble AUC
* Interactive visualization

### 🔹 Batch Processing Mode

* Upload CSV or Excel files
* Run all scenario-policy combinations
* Export results as a consolidated Excel file

---

## ✨ Key Features

* Ensemble-based prediction and uncertainty estimation
* Scenario-policy simulation framework
* Interactive web interface using Streamlit
* Batch processing capability
* Data export to Excel
* Visualization using Plotly

---

## 📌 Notes

* All trained models are stored in the `Ensemble_Committee_Models` directory
* The app uses **relative paths**, making it compatible with cloud deployment
* Ensure all dependencies in `requirements.txt` are installed before running locally

---

## 👨‍💻 Author

**Vali Vakhshoori**
PhD in Geography-GIS
Machine Learning | Spatial Analysis | Agricultural Systems Modeling

---

## 📄 License

This project is intended for research and academic purposes.
