# uoa-p2-engineering-mingpa-predictor
Supervised machine learning model to predict GPA cutoffs for University of Auckland engineering specialisations using decision trees.


# 🧠 UoA Part II Engineering GPA Cutoff Predictor

A supervised machine learning tool to estimate GPA cutoffs for entry into University of Auckland's Part II engineering specialisations based on historical data, popularity scores, and cohort size.

---

## 📌 Project Objective

The goal of this project is to help students predict the minimum GPA required for their desired engineering specialisation using past data and a machine learning model. This can help manage expectations and inform academic planning for students.

---

## 🔍 How It Works

- Trained on historical data from **2019 to 2025**
- Users provide:
  - Estimated **popularity score** (1–10) for each spec
  - Expected **cohort size**
- Predicts **minimum GPA cutoffs for 2026/any other cohort** based on user input on popularity and cohort size and makes use of trained data from 2019-2025 cohorts. 
- A CLI interface displays predicted GPA ranges and flags a spec as "N/A" if demand is too low

---

## 🧠 Machine Learning Model

- **Type:** Supervised Regression
- **Model Used:** `DecisionTreeRegressor` from `scikit-learn`
- **Features Used:**
  - Year
  - Cohort Size
  - Seats Available
  - Current and Previous Popularity Scores
  - Previous Year GPA
  - Missing Value Flags
- **Target:** `LowerBoundGPA` (predicted GPA cutoff)

---

## 🛠 Technologies

- `Python 3`
- `pandas` – Data manipulation
- `numpy` – Numerical processing
- `scikit-learn` – Machine learning model
- `tabulate` – Formatted CLI output

---

## 💻 How to Run

1. Clone this repository or download the `.py` file if you have the libraries (pandas, numpy, scikit-learn, tabulate) pre installed
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python gpa_predictor.py
```

4. Follow the CLI prompts to enter:
   - Total cohort size (e.g. 987)
   - Popularity scores for each specialisation

---

## ✨ Example Output

```
--- Predicted GPA Cutoffs for 2026 ---
Specialisation           Lower GPA    Upper GPA
------------------------------------------------
Biomedical               3.56         3.76
Mechanical               6.90         7.10
Civil & Env              3.00         3.20
Structural               N/A          N/A
...
```

Note: `'N/A'` indicates a specialisation may not be filled due to low popularity and demand.

---

## 📈 Why Decision Trees?

- Handles non-linear relationships in GPA vs. popularity/seats
- Works well with small, structured datasets
- Easy to understand and explain to non-technical users

---

## 📁 Files

- `gpa_predictor.py`: Main script
- `requirements.txt`: Dependency list
- `README.md`: You're reading it 🙂

## 👨‍💻 Author

Made with 💻 and 📊 by Ashmit Bhola – feel free to connect on [LinkedIn](https://www.linkedin.com/in/ashmit-bhola/) 

---
