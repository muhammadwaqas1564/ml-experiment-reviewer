"""Customer churn model - CORRECT VERSION of leaky_churn.py.

Same synthetic data, no leakage. mlreview should report no issues here.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

features, labels = make_classification(n_samples=4000, n_features=12, n_informative=4,
                                       weights=[0.95], flip_y=0.03, random_state=0)
df = pd.DataFrame(features, columns=[f"f{i}" for i in range(12)])
df["churn"] = labels
df.loc[df.sample(frac=0.05, random_state=1).index, "f0"] = np.nan

y = df["churn"]
X = df.drop(columns=["churn"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

model = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED),
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, model.predict(X_test), digits=3))
print(f"test ROC AUC          : {roc_auc_score(y_test, proba):.3f}")
print(f"test average precision: {average_precision_score(y_test, proba):.3f}")
