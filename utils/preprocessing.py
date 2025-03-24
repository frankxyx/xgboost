# utils/preprocessing.py
import pandas as pd

def load_preprocess(path='merged_df.csv'):
    """
    Loads the merged dataset, detects and casts categorical features,
    and returns processed features (X), target (y), and list of categorical columns.
    """

    # Load the merged dataset
    df = pd.read_csv(path)

    # 🔻 Drop any problematic or irrelevant object columns (like VIN)
    if 'VIN' in df.columns:
        df = df.drop(columns=['VIN'])

    # Columns to exclude from feature detection
    exclude_cols = ['CASENUM', 'PERNO', 'VEHNO', 'INJSEV_H']
    likely_categoricals = []

    # Detect categorical features based on integer type + few unique values
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() < 15:
            likely_categoricals.append(col)

    # (Optional) If you're NOT using enable_categorical=True, skip this casting
    # for col in likely_categoricals:
    #     df[col] = df[col].astype("category")

    print("✅ Categorical features detected:", likely_categoricals)

    # Target
    y = df['INJSEV_H']
    X = df.drop(columns=exclude_cols)

    return X, y, likely_categoricals
