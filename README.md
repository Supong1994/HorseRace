# HorseRace
The goal is to predict the outcome of horse races — specifically, whether a horse wins or places — using machine learning models (Random Forest, XGBoost, LightGBM) trained on historical race data.

🔁 Pipeline Steps (Code Logic)
### 🧩 1. Data Loading & Cleaning
Files:

horses_2020.csv – data about horses in races

races_2020.csv – data about each race

Cleaning:

Columns with mixed numeric and string values (weightSt, decimalPrice, etc.) are cleaned using clean_numeric().

Dates and times are parsed correctly using pd.to_datetime.

✔ Merged using rid (race ID) into one merged_df.

🏗 2. Feature Engineering
Enhances the dataset with meaningful features:

Date Features: year, month, weekday, hour.

Weight Feature: total weight carried by the horse.

Odds Features: log_price, price_rank.

Position Normalization: how well a horse performed relative to others in the same race.

Historical Stats:

Jockey: win and place rates.

Trainer: win and place rates.

Horse: average performance and race counts.

✅ These help the model learn patterns from past performance.

🎯 3. Target Definition
Defines what we're predicting:
'target' = 1 if horse won the race (position == 1), else 0.

🟡 Option to switch to place prediction (position <= 3) exists via target_type.

🧪 4. Temporal Data Split
Data is split chronologically to avoid data leakage:

80% for training

20% for testing

Ensures model learns from past to predict future races.

🧠 5. Feature Selection
Separates numeric and categorical features:

Numeric: ages, ratings, weights, engineered stats

Categorical: course, trainerName, jockeyName

Only keeps features present in the dataset.

🔄 6. Preprocessing Pipeline
Numeric: median imputation + standard scaling

Categorical: fill missing with "missing" + OneHotEncoding (sparse)

🔧 Uses ColumnTransformer to automatically handle all columns during training.

🤖 7. Model Training
Trains three models:

Random Forest

XGBoost

LightGBM

Each is trained using:

A Pipeline that includes preprocessing + model

Balanced class weights and reduced estimators for speed

Models are stored in a dictionary trained_models.

📊 8. Model Evaluation
Each model is evaluated on the test set:

Predictions (y_pred) and probabilities (y_pred_proba)

Metrics computed:

accuracy

log_loss (lower is better)

roc_auc (higher is better)

classification_report (precision, recall, f1-score)

Then selects the best model (lowest log loss), and saves it.

