# main.py

from utils.preprocessing import load_preprocess
from models.xgb_model import train_xgb_model
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score

# Step 1: Load and preprocess data
X, y, categorical_columns = load_preprocess()

# Step 2: Train the XGBoost model
model = train_xgb_model(X, y)

# Step 3: Evaluate using stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1 = make_scorer(f1_score, average='weighted')

print("🔁 Running 5-fold stratified cross-validation...")
scores = cross_val_score(model, X, y, cv=cv, scoring=f1)

print(f"✅ Cross-validated weighted F1 scores: {scores}")
print(f"📊 Mean F1 Score: {scores.mean():.4f}")
