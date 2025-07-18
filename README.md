# UoA Engineering Specialisation GPA cut-off Predictor

https://gpa-predictor-web-app.streamlit.app/ 
An **interactive Streamlit web app** that predicts GPA cutoffs for entry into engineering specialisations at the University of Auckland.

It allows you to:
- Explore **historical patterns** (2019–2025) between intake data, grade distributions, and actual GPA cutoffs.
- Experiment with different **machine learning models** that you can choose yourself.
- Enter hypothetical **future scenarios (e.g., 2026)** and see predicted GPA cutoffs.

## ✨ Features

**Historic Data Exploration**  
- Adjust input variables like Seats Available, Cohort Size, Popularity Score, and average grade statistics.  
- See what GPA cutoff the model would have predicted historically for those conditions.  
- View **model performance metrics** (MAE, RMSE, R²) and an **Actual vs. Predicted** plot.  
- Inspect **feature importance** to see which inputs the model relied on.  
- **Choose the model** yourself: Decision Tree, Random Forest, Linear Regression, Gradient Boosting, or XGBoost.

**Future Prediction (e.g., 2026)**  
- Enter your own assumptions for seats, popularity, cohort size, and grade distributions.
- Predict a future GPA cutoff based on patterns learned from historical data.
- Validation checks ensure grade distributions make sense before prediction.

**Multiple Models to Compare**  
When exploring data or predicting, you can switch between:
- Decision Tree
- Random Forest
- Linear Regression
- Gradient Boosting
- XGBoost  
…and immediately see how their predictions and metrics differ.

**Built‑in Validations**  
- Ensures `A% + B% + C% = 100%` (all grade bands accounted for).  
- Ensures `C% (including fails) ≥ FailRate (100 − PassRate)` for logical consistency.

## 📅 How Historical Prediction Works

The app has historical records from:
- **2019–2025 cohorts** (specialisation intake data)
- Merged with **2018–2024 course grade distributions**.

Each record corresponds to **one specialisation in one year**, with features such as:
| Feature Name       | Description                                                                                   | Example Values |
|-------------------|-----------------------------------------------------------------------------------------------|----------------|
| **SeatsAvailable** | Number of seats offered in that specialisation for the year.                                 | 35, 120, 185   |
| **CohortSize**     | Total number of students in the cohort applying that year.                                   | 875, 1020, 1100 |
| **PopularityScore**| Relative popularity of the specialisation (based on student rankings/preferences).           | 2.0, 7.0, 10.0 |
| **AvgCourseGPA**   | Average GPA across all relevant courses for the previous year.                               | 4.8, 6.0, 7.2 |
| **MedianCourseGPA**| Median GPA across all relevant courses for the previous year.                                | 4.5, 5.8, 6.4 |
| **PassRate**       | Percentage of students passing courses (C and above).                                        | 85%, 90%, 95% |
| **PctA**           | Percentage of students achieving an A‑range grade (A+, A, A‑).                               | 30%, 45%, 60% |
| **PctB**           | Percentage of students achieving a B‑range grade (B+, B, B‑).                               | 20%, 35%, 50% |
| **PctCOrLower**    | Percentage of students receiving C grades or lower (C+, C, C‑, D+, D, D‑).                   | 10%, 25%, 40% |
| **VarGPA**         | Variance of course GPA across all courses (higher = more variation in student performance).  | 0.8, 1.2, 2.0 |
| **Specialisation** | One‑hot encoded indicator columns for each engineering specialisation (e.g., Software, Civil). | 0 or 1 values |

**Important:**  
The model does **not use the year as a feature**.  
It only uses those input variables to learn relationships.

### Training and Evaluation
1. The dataset (2019–2025 rows) is split into:
   - **Train set (80%)** – the model learns patterns from these records.
   - **Test set (20%)** – used to check how well the model generalizes.

2. After training, the app:
   - Predicts cutoffs for the test set (which correspond to actual past years and specialisations, e.g., *2021 Software*, *2023 Civil*).
   - Compares predicted cutoffs to the known historical cutoffs for those same records.
   - Shows you **MAE, RMSE, and R²** and an **Actual vs. Predicted plot** so you can judge model performance.

**So when you explore historical data:**  
👉 You’re not asking for a specific year.  
👉 You’re seeing how well the model would have predicted the known cutoffs (2019–2025) based on the conditions in those records.



## 📈 Validations

Before making a future prediction, the app checks:
- **Grade bands completeness:** `A% + B% + C%` must equal 100%.  
- **Logical consistency:** `C% (including fails) >= FailRate (100 − PassRate)`.  
If these conditions aren’t met, the app displays an error and stops prediction.

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Install dependencies:
```bash
pip install -r requirements.txt
```

(Mac users with XGBoost:)
Install OpenMP runtime if you see `libomp` errors:
```bash
brew install libomp
```
### Running the app
In your terminal:
```bash
streamlit run streamlit_app.py
```
Then open the local URL it prints, usually:
```bash
http://localhost:8501
```
## Project Structure
```bash
.
├── uoa_engineering_gpa_data.csv            # Historical intake data
├── grade distribution - 2018 to 2024.xlsx  # Historical grade distributions
├── streamlit_app.py                        # Main app code
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```
## 🌐 Live Demo 

👉 **Streamlit Cloud Link:**   https://gpa-predictor-web-app.streamlit.app/

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io)
- **Backend:** Python 3, pandas, scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn

## 📌 Notes

- The model is trained on historical data with limited scope, so predictions are for **exploration only** — not official cutoffs.
- You can **choose the model** in the app to see how different approaches (Decision Tree, Random Forest, Linear Regression, Gradient Boosting, XGBoost) behave on the same data.
- The app is open source — feel free to fork, experiment, or extend it!





