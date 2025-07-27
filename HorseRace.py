import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_and_preprocess_data():
    # Load datasets
    horse_df = pd.read_csv('D:\\Datasets\\HorseRace\\horses_2020.csv', low_memory=False)
    race_df = pd.read_csv('D:\\Datasets\\HorseRace\\races_2020.csv', low_memory=False)

    # Merge datasets
    merged_df = pd.merge(horse_df, race_df, on='rid', how='inner')

    # Convert date with error handling
    merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')

    # Create date-related features
    date_features = ['race_year', 'race_month', 'race_day_of_week']
    for i, attr in enumerate(['year', 'month', 'dayofweek']):
        merged_df[date_features[i]] = getattr(merged_df['date'].dt, attr)

    # Fill missing date features with most common value
    for col in date_features:
        mode_val = merged_df[col].mode()
        if not mode_val.empty:
            merged_df[col] = merged_df[col].fillna(mode_val.iloc[0])
        else:
            merged_df[col] = merged_df[col].fillna(0)  # fallback

    # Parse distance (handle miles and furlongs)
    def parse_distance(d):
        if pd.isna(d):
            return np.nan
        try:
            if isinstance(d, str):
                d = d.lower().strip()
                if 'm' in d and 'f' in d:
                    miles = float(d.split('m')[0])
                    furlongs = float(d.split('m')[1].split('f')[0])
                    return miles * 8 + furlongs
                elif 'm' in d:
                    return float(d.split('m')[0]) * 8
                elif 'f' in d:
                    return float(d.split('f')[0])
            return float(d)
        except:
            return np.nan

    merged_df['distance_furlongs'] = merged_df['distance'].apply(parse_distance)

    # Convert time to seconds
    def time_to_seconds(t):
        if pd.isna(t):
            return np.nan
        try:
            if isinstance(t, str):
                parts = t.replace(',', '.').split(':')
                if len(parts) == 3:  # HH:MM:SS
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                elif len(parts) == 2:  # MM:SS
                    return int(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 1:  # SS
                    return float(parts[0])
            return float(t)
        except:
            return np.nan

    merged_df['time_seconds'] = merged_df['time'].apply(time_to_seconds)

    # Clean prize money (fixed regex escape)
    merged_df['prize'] = merged_df['prize'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    merged_df['prize'] = pd.to_numeric(merged_df['prize'], errors='coerce').fillna(0)

    # Create target variable
    merged_df['winner'] = (merged_df['position'] == 1).astype(int)

    # Drop unnecessary columns (but keep horseName for individual predictions)
    cols_to_drop = [
        'outHandicap', 'overWeight', 'hurdles', 'band', 'headGear',
        'rclass', 'currency', 'price', 'dist', 'positionL', 'gfather',
        'trainerName', 'saddle', 'jockeyName', 'mother', 'time', 'date',
        'distance', 'rid', 'course', 'title', 'father'
    ]
    merged_df = merged_df.drop(columns=[col for col in cols_to_drop if col in merged_df.columns])

    return merged_df, horse_df  # Return both merged data and original horse data


def build_model_pipeline():
    # Define feature sets
    numeric_features = [
        'age', 'decimalPrice', 'weightLb', 'RPR', 'TR', 'OR',
        'runners', 'margin', 'distance_furlongs', 'prize',
        'race_year', 'race_month', 'race_day_of_week', 'time_seconds'
    ]

    categorical_features = ['condition', 'class']

    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            max_depth=10,
            min_samples_leaf=5))
    ])

    return pipeline


def predict_horse_win_probability(model, horse_data, features, horse_name):
    """Predict win probability for a specific horse"""
    # Filter data for the specific horse
    horse_data = horse_data[horse_data['horseName'] == horse_name].copy()

    if horse_data.empty:
        raise ValueError(f"No data found for horse: {horse_name}")

    # Prepare the input data
    X = horse_data[features]

    # Predict probabilities
    proba = model.predict_proba(X)[:, 1]  # Probability of class 1 (win)

    # Add predictions to the dataframe
    horse_data['win_probability'] = proba
    horse_data['predicted_win'] = (proba >= 0.5).astype(int)

    return horse_data[['horseName', 'race_year', 'race_month', 'decimalPrice',
                       'RPR', 'TR', 'OR', 'win_probability', 'predicted_win', 'winner']]


def main():
    # Load and preprocess data (keeping horse names)
    df, original_horse_df = load_and_preprocess_data()

    # Prepare features and target
    features = ['age', 'decimalPrice', 'weightLb', 'RPR', 'TR', 'OR',
                'runners', 'margin', 'distance_furlongs', 'prize',
                'race_year', 'race_month', 'race_day_of_week', 'time_seconds',
                'condition', 'class']
    X = df[features]
    y = df['winner']

    # Remove rows with missing target
    X = X[~y.isna()]
    y = y[~y.isna()]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Build and train model
    model = build_model_pipeline()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Feature importance
    try:
        # Get feature names
        numeric_features = model.named_steps['preprocessor'].transformers_[0][2]
        ohe = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        cat_features = model.named_steps['preprocessor'].transformers_[1][2]

        feature_names = numeric_features.copy()
        for i, col in enumerate(cat_features):
            categories = ohe.categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])

        # Get importances
        importances = model.named_steps['classifier'].feature_importances_

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nTop 20 most important features:")
        print(importance_df.head(20))

    except Exception as e:
        print(f"\nFeature importance extraction failed: {str(e)}")

    # Predict for specific horse (e.g., "George Gently")
    try:
        horse_name = "Waterproof"  # Change to any horse name
        predictions = predict_horse_win_probability(model, df, features, horse_name)

        print(f"\nPredictions for {horse_name}:")
        print(predictions.sort_values('win_probability', ascending=False).head(10))

        if not predictions.empty:
            # Get most recent race prediction
            latest_race = predictions.sort_values(['race_year', 'race_month'], ascending=False).iloc[0]
            print(f"\nMost recent race prediction for {horse_name}:")
            print(f"Win Probability: {latest_race['win_probability']:.2%}")
            print(f"Predicted to win: {'Yes' if latest_race['predicted_win'] else 'No'}")
            print(f"Actual result: {'Won' if latest_race['winner'] == 1 else 'Lost'}")

    except ValueError as e:
        print(f"\nPrediction error: {str(e)}")


if __name__ == "__main__":
    main()