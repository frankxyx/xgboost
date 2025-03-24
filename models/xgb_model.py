# models/xgb_model.py

from xgboost import XGBClassifier

def train_xgb_model(X, y, num_classes=7, random_state=42):
    """
    Create and train an XGBoost classifier for multiclass classification.

    Parameters:
    - X: pd.DataFrame, input features
    - y: pd.Series or array-like, target labels
    - num_classes: int, number of target classes
    - random_state: int, for reproducibility

    Returns:
    - Trained XGBClassifier model instance
    """
    model = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        enable_categorical=True,
        tree_method='hist',
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=1,
        random_state=random_state
    )
    model.fit(X, y)
    return model
