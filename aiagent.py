import numpy as np
import pandas as pd
import joblib

import pandas as pd

file_path = '/content/AI_Agent_Adoption_Workplace_Productivity_2024_2026.csv'
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
display(df.head())
print(df.info())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("/content/logistic_regression_model.joblib")


# ==========================================
# 2. Identify Target and Feature Columns
# ==========================================

target_candidates = [
    col for col in df.columns
    if any(keyword in col.lower()
           for keyword in ['adoption', 'status', 'productive'])
]

target_col = target_candidates[0] if target_candidates else df.columns[-1]

print(f"Selected Target Column: {target_col}")


# ==========================================
# 3. Separate Features and Target
# ==========================================

X = df.drop(columns=[target_col])
y = df[target_col]


# ==========================================
# 4. Identify Numerical and Categorical Features
# ==========================================

num_cols = X.select_dtypes(include=['number']).columns.tolist()

cat_cols = X.select_dtypes(
    include=['object', 'category', 'bool']
).columns.tolist()


# ==========================================
# 5. Create Preprocessing Pipelines
# ==========================================

# Numerical preprocessing
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# Categorical preprocessing
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)


# ==========================================
# 6. Build Machine Learning Pipeline
# ==========================================

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


# ==========================================
# 7. Split Dataset into Training and Testing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 8. Train the Model
# ==========================================

model_pipeline.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")


# ==========================================
# 9. Make Predictions
# ==========================================

y_pred = model_pipeline.predict(X_test)


# ==========================================
# 10. Evaluate Model Performance
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 11. Save Trained Model
# ==========================================

joblib.dump(model_pipeline, "trained_model.pkl")

print("\nModel saved successfully as trained_model.pkl")


# ==========================================
# 12. Main Execution
# ==========================================

if __name__ == "__main__":
    print("\nMachine Learning Application Executed Successfully!")
