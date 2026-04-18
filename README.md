# AgriTech-Adoption-App
Streamlit app for agri-tech policy simulation

A Streamlit-based machine learning application that simulates agricultural technology adoption under different policy, scenario, and technology configurations. The system uses an **ensemble of Random Forest models** to estimate adoption probabilities and uncertainty across multiple socioeconomic and policy settings.

---

## 🚀 Live App
Once deployed, your app will be available here:  
👉 https://your-app-name.streamlit.app

---

## 📊 Project Overview

This tool allows users to:
- Simulate **agri-technology adoption likelihood**
- Evaluate impacts of different:
  - Policies (Financial, Education, R&D, etc.)
  - Production scenarios (Increase / Decrease)
  - Technologies (Digital, CEA, Genomic)
- Input farmer socioeconomic profiles
- Run **ensemble predictions using multiple trained models**
- Perform **batch analysis for multiple farmer records**
- Visualize adoption probability distributions

---

## 🧠 Machine Learning Approach

- Algorithm: **Random Forest (Scikit-learn)**
- Ensemble structure: Multiple trained models per scenario-policy combination
- Outputs:
  - Mean adoption probability
  - Min–Max uncertainty range
  - Ensemble AUC score

---

## ⚙️ Installation (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/VVakhshoori/AgriTech_Adoption_App.git
cd AgriTech_Adoption_App
2. Install dependencies
pip install -r requirements.txt
3. Run the app
AgriTech_Adoption_App.py

📦 Requirements
streamlit
pandas
numpy
scikit-learn
joblib
plotly
xlsxwriter
openpyxl

📥 Input Features

The model uses the following socioeconomic variables:

Age
Total farm area
Employee number
Farm revenue
Asset value
Hours on/off farm
Education level
Gender
Immigration status
Indigenous status

📊 Outputs
Single Prediction Mode
Adoption probability
Adoption category (Low / Moderate / Strong)
Probability range across models
Ensemble AUC
Interactive bar chart of model outputs
Batch Mode
Upload CSV/XLSX files
Run 16 scenario-policy combinations
Export combined Excel results

🧪 Key Features
Ensemble-based uncertainty estimation
Scenario-policy simulation framework
Interactive Streamlit dashboard
Batch processing for multiple records
Excel export functionality
Plotly-based visualization

📌 Notes
Models must be stored in the Ensemble_Committee_Models directory
The app uses cached model loading for performance optimization
Ensure all required dependencies are installed before running

👨‍💻 Author

Developed by: Vali Vakhshoori
PhD in Geography-GIS | Machine Learning | Spatial Analysis | Agricultural Systems Modeling
