import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor  # XGBoost model

# Set font to Arial for consistency and readability
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cache the data loading and preprocessing so it only happens once unless files change
@st.cache_data
def load_and_prepare_data():
    # Load main GPA cutoff dataset
    raw = pd.read_csv("uoa_engineering_gpa_data.csv")
    # Load grade distribution data
    grades = pd.read_excel("grade distribution - 2018 to 2024.xlsx", sheet_name="Sheet1")
    grades = grades.dropna(subset=["Year"])
    grades["Year"] = grades["Year"].astype(int)
    grades["CohortYear"] = grades["Year"] + 1

    # Define grade points for weighted GPA calculations
    grade_points = {
        "A+": 9, "A": 8, "A-": 7,
        "B+": 6, "B": 5, "B-": 4,
        "C+": 3, "C": 2, "C-": 1,
        "D+": 0, "D": 0, "D-": 0
    }

    # Ensure all expected grade columns exist
    for g in grade_points:
        if g not in grades.columns:
            raise ValueError(f"Missing grade column: {g}")

    # Fill missing grade counts with zeros
    for g in grade_points:
        grades[g] = grades[g].fillna(0)

    # Calculate total grades and weighted GPA sum
    grades["TotalGrades"] = sum(grades[g] for g in grade_points)
    grades["WeightedSum"] = sum(grades[g] * p for g, p in grade_points.items())

    # Remove rows where no valid grades were recorded
    grades = grades[grades["TotalGrades"] > 0]

    # Calculate GPA-related metrics for each course row
    def row_median(row):
        values = []
        for g, p in grade_points.items():
            values.extend([p] * int(row[g]))
        if not values:
            return np.nan
        return float(np.median(values))

    grades["AvgCourseGPA_row"] = grades["WeightedSum"] / grades["TotalGrades"]
    grades["PassRate_row"] = grades["Successful Completions"] / grades["Course Headcount"]
    grades["PctA_row"] = (grades["A+"] + grades["A"] + grades["A-"]) / grades["TotalGrades"]
    grades["PctB_row"] = (grades["B+"] + grades["B"] + grades["B-"]) / grades["TotalGrades"]
    grades["PctCOrLower_row"] = (grades["C+"] + grades["C"] + grades["C-"] +
                                 grades["D+"] + grades["D"] + grades["D-"]) / grades["TotalGrades"]
    grades["MedianCourseGPA_row"] = grades.apply(row_median, axis=1)

    # Aggregate metrics by cohort year
    var_by_year = grades.groupby("CohortYear")["AvgCourseGPA_row"].var().reset_index()
    var_by_year.rename(columns={"AvgCourseGPA_row": "VarGPA"}, inplace=True)
    agg = grades.groupby("CohortYear").agg(
        AvgCourseGPA=("AvgCourseGPA_row", "mean"),
        MedianCourseGPA=("MedianCourseGPA_row", "mean"),
        PassRate=("PassRate_row", "mean"),
        PctA=("PctA_row", "mean"),
        PctB=("PctB_row", "mean"),
        PctCOrLower=("PctCOrLower_row", "mean")
    ).reset_index()
    agg = agg.merge(var_by_year, on="CohortYear", how="left")

    # Merge aggregated grade features into main dataset
    raw = raw.merge(agg, left_on="Year", right_on="CohortYear", how="left").drop(columns=["CohortYear"])

    # Drop rows without target GPA cutoff and convert specialisation to dummy variables
    data = raw.dropna(subset=["LowerBoundGPA"])
    data = pd.get_dummies(data, columns=["Specialisation"], drop_first=True)

    return data

# Cache model training so models are trained only once
@st.cache_resource
def train_all_models(X_train, y_train):
    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# Load and prepare data
data = load_and_prepare_data()

# Separate features and target
y = data["LowerBoundGPA"]
X = data.drop(columns=["LowerBoundGPA", "Year"])

# Split into train and test sets for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train all models (cached)
models = train_all_models(X_train, y_train)

# Title and introduction text
st.title("UoA Engineering GPA Predictor")
st.markdown("""
This tool predicts minimum GPA cutoffs for engineering specialisations at the University of Auckland.

Enter your assumptions for a future year (e.g. 2026) — seats, cohort size, popularity,
and various grade-distribution metrics — to obtain a predicted cutoff. The model is
trained on historic data but you won't need to interact with it directly.

In the back end, we load the cutoff's and grade‑distribution data from 2019–2025, compute
aggregate features by cohort (average/median GPA, pass rates, grade shares, GPA
variance, etc.), and fit several regression learners (decision tree, random forest,
gradient boosting, XGBoost and linear regression). The year itself is not used as a
feature so the learned relationships are purely driven by the input conditions.

When a prediction is requested the selected model returns a point estimate; the
quoted error margin (±) is the model's root‑mean‑squared error on the fixed 20 %
hold‑out test set from the historic data. However, a single train/test split can
be sensitive to which data happens to land in each set, so we also compute 
five‑fold cross‑validation across the entire dataset. In cross‑validation, the data 
is randomly divided into 5 equal chunks (folds); the model is then trained 5 times, 
each time leaving a different fold out for testing and training on the other 4. 
The final metrics (MAE, RMSE, R²) are averaged across all 5 folds. This approach 
is much more robust because every data point is used both for training and testing, 
and the variation in metrics across folds gives a sense of how sensitive the model 
is to the specific split. Both sets of metrics are reported after each prediction 
so you can compare the two evaluation methods and get a gist of how accurate the prediction is.

Features used in the model:
- SeatsAvailable, CohortSize, PopularityScore
- AvgCourseGPA, MedianCourseGPA, PassRate, PctA, PctB, PctCOrLower, VarGPA
""")

# Historical data section
# (removed – only forward‑looking predictions are shown to avoid confusion)
# Future prediction section
st.header("Predict 2026 Requirement")
st.markdown("""
- Enter your assumptions for 2026, including seats available, cohort size,
  popularity score and grade-distribution percentages.
- Grade-related numbers (pass rate, %A, %B, %C-or-lower) are entered as whole
  percentages for clarity; they are converted to proportions internally.
- After you click **Predict 2026 GPA** the selected model will produce a point
  estimate and display an approximate error margin (±RMSE) derived from historical
  performance.
- The app evaluates each model two ways: using the 20 % hold‑out test set and via
  5‑fold cross‑validation on the whole dataset. Both sets of MAE/RMSE/R² metrics
  are shown so you can see the difference.
- Look at the performance metrics below (MAE, RMSE, R²) to understand how the
  model has fared on past data; smaller errors and higher R² indicate more
  reliable predictions.
""")

spec_2026 = st.selectbox(
    "Select Specialisation for 2026 Prediction",
    sorted([c for c in data.columns if c.startswith("Specialisation_")]),
    key="future_spec"
)

# User inputs for future scenario
seats_2026 = st.number_input("Seats Available in 2026", min_value=30, max_value=250, value=100, key="future_seats")
cohort_2026 = st.number_input("Cohort Size in 2026", min_value=800, max_value=1500, value=1000, key="future_cohort")
popularity_2026 = st.number_input("Popularity Score in 2026", min_value=1.0, max_value=10.0, value=5.0, step=0.1, key="future_popularity")
avg_gpa_2026 = st.slider("Assumed AvgCourseGPA in 2026", 0.0, 9.0, float(data["AvgCourseGPA"].mean()), step=0.1, key="future_avg_gpa")
median_gpa_2026 = st.slider("Assumed MedianCourseGPA in 2026", 0.0, 9.0, float(data["MedianCourseGPA"].mean()), step=0.1, key="future_median_gpa")
pass_rate_2026 = st.number_input("Assumed PassRate in 2026 (%)", 0, 100, int(data["PassRate"].mean() * 100), step=1, key="future_passrate")
pct_a_2026 = st.number_input("Assumed % of A-range grades in 2026", 0, 100, int(data["PctA"].mean() * 100), step=1, key="future_pcta")
pct_b_2026 = st.number_input("Assumed % of B-range grades in 2026", 0, 100, int(data["PctB"].mean() * 100), step=1, key="future_pctb")
pct_c_2026 = st.number_input("Assumed % of C or lower grades in 2026", 0, 100, int(data["PctCOrLower"].mean() * 100), step=1, key="future_pctc")
var_gpa_2026 = st.slider("Assumed VarGPA in 2026", 0.0, 5.0, float(data["VarGPA"].mean()), step=0.1, key="future_vargpa")

# Select model for future prediction
model_choice_2026 = st.selectbox("Choose model for 2026 prediction:", list(models.keys()), key="future_model")
model_2026 = models[model_choice_2026]

# Handle predict button
if st.button("Predict 2026 GPA", key="predict_future"):
    error_msgs = []

    # Calculate total of A+B+C
    sum_abc = pct_a_2026 + pct_b_2026 + pct_c_2026
    fail_rate = 100 - pass_rate_2026

    # Check that A+B+C equals 100%
    if abs(sum_abc - 100) > 0.5:  # allow small rounding differences
        error_msgs.append(
            f"A-range ({pct_a_2026:.1f}%) + B-range ({pct_b_2026:.1f}%) + C-range ({pct_c_2026:.1f}%) = {sum_abc:.1f}%. "
            "These must add up to 100%."
        )
        # Check that C-or-lower% is at least fail rate
    if pct_c_2026 + 0.5 < fail_rate:  # allow small tolerance
        error_msgs.append(
            f"C-range (including D or lower) is {pct_c_2026:.1f}%, but fail rate (100 - PassRate) is {fail_rate:.1f}%. "
            "C-range (including fails) must be at least as large as the fail rate, because all failing students are within C or lower."
        )

    # Show errors if any
    if error_msgs:
        for m in error_msgs:
            st.error(m)
        st.stop()
    # Prepare input row for future scenario
    input_dict_2026 = {
        "SeatsAvailable": seats_2026,
        "CohortSize": cohort_2026,
        "PopularityScore": popularity_2026,
        "AvgCourseGPA": avg_gpa_2026,
        "MedianCourseGPA": median_gpa_2026,
        "PassRate": pass_rate_2026 / 100.0,
        "PctA": pct_a_2026 / 100.0,
        "PctB": pct_b_2026 / 100.0,
        "PctCOrLower": pct_c_2026 / 100.0,
        "VarGPA": var_gpa_2026
    }
    for col in data.columns:
        if col.startswith("Specialisation_"):
            input_dict_2026[col] = 1 if col == spec_2026 else 0
    input_df_2026 = pd.DataFrame([input_dict_2026])

    # Generate prediction for future scenario
    predicted_gpa_2026 = model_2026.predict(input_df_2026)[0]

    # Evaluate historical test performance so we can quote an error margin
    y_pred_test_2026 = model_2026.predict(X_test)
    mae_26 = mean_absolute_error(y_test, y_pred_test_2026)
    rmse_26 = np.sqrt(mean_squared_error(y_test, y_pred_test_2026))
    r2_26 = r2_score(y_test, y_pred_test_2026)

    # also compute 5-fold cross-validation on the full dataset
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(
        model_2026, X, y,
        scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'],
        cv=5
    )
    cv_mae = -cv_results['test_neg_mean_absolute_error'].mean()
    cv_rmse = -cv_results['test_neg_root_mean_squared_error'].mean()
    cv_r2 = cv_results['test_r2'].mean()

    # Display prediction with approximate ±RMSE margin (hold-out)
    st.success(
        f"Predicted 2026 GPA cutoff for {spec_2026.replace('Specialisation_', '')}: "
        f"{predicted_gpa_2026:.2f} ± {rmse_26:.2f} (approx.)"
    )

    st.markdown("Model performance on past data (20% hold-out):")
    st.write(f"MAE: {mae_26:.3f} | RMSE: {rmse_26:.3f} | R²: {r2_26:.3f}")
    st.markdown("5‑fold cross‑validation (entire dataset):")
    st.write(f"MAE: {cv_mae:.3f} | RMSE: {cv_rmse:.3f} | R²: {cv_r2:.3f}")

    with st.expander("How to interpret these metrics"):
        st.markdown("""
**Mean Absolute Error (MAE)** – the average absolute difference between the
predicted cutoff and the actual historical cutoff. Smaller values mean the model's
predictions are, on average, closer to reality.

**Root Mean Squared Error (RMSE)** – like MAE but squaring the errors before
averaging gives extra penalty to larger misses; it approximates the standard
deviation of the prediction errors and is the quantity shown as the ± margin.

**R² (coefficient of determination)** – the fraction of variance in the cutoff
that the model explains. A value of 1.0 means perfect predictions; 0 indicates the
model is no better than always guessing the average cutoff.

**Hold‑out test set** – a fixed 20 % slice of the historical data that was held back
before training. Metrics here show how the trained model performs on unseen cases
and can vary depending on which slice is chosen.

**Cross‑validation** – the data is split into 5 different training/test folds and the
model is evaluated on each. The reported CV metrics are the averages across folds
and typically provide a more stable, robust estimate of generalisation
performance.
""")

    # Show feature importances
    if hasattr(model_2026, "feature_importances_"):
        st.subheader("Which factors mattered the most?")
        st.markdown(
            """Feature importances are a way to measure how much each input variable (feature) contributed to the model's predictions. 
For tree‑based models (like random forest, gradient boosting, XGBoost), this is provided by the model's `feature_importances_` attribute—a built-in property that automatically calculates and stores these importance values after training. 
Each time a feature is used to split the data in a tree, the model tracks how much that split reduces prediction error (such as mean squared error). The final importance for each feature is the total reduction in error it provided, averaged over all trees. 
A higher value means the model relied more on that feature when making predictions.

Linear Regression works differently — it fits a single equation with coefficients for each feature. The magnitude of a coefficient (especially when features are normalized) indicates how strongly that feature influences the prediction. However, linear models don't have a built‑in `feature_importances_` attribute, so they won't show a chart. Tree‑based models are generally better for inspection since they expose importances directly.
"""
        )
        fi_df = pd.DataFrame({"Feature": X.columns, "Importance": model_2026.feature_importances_})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(10)
        fig3, ax3 = plt.subplots(figsize=(8,6))
        sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax3)
        ax3.set_title(f"Top 10 Feature Importances ({model_choice_2026})")
        st.pyplot(fig3)

        with st.expander("How to interpret this bar chart"):
            st.markdown("""
Each bar shows how important that feature was in making predictions. Longer bars mean the feature was more important to the model.
""")
