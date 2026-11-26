"""Customer churn model - DEMO WITH HIDDEN BUGS.

Runs on synthetic data (no download). Compare its scores with clean_churn.py,
then run:  python mlreview.py examples/leaky_churn.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# synthetic churn data: about 5% of customers churn
features, labels = make_classification(n_samples=4000, n_features=12, n_informative=4,
                                       weights=[0.95], flip_y=0.03, random_state=0)
df = pd.DataFrame(features, columns=[f"f{i}" for i in range(12)])
df["churn"] = labels
df.loc[df.sample(frac=0.05, random_state=1).index, "f0"] = np.nan

df = df.fillna(df.mean())

y = df["churn"]
X = df.drop(columns=["f11"])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

print(f"train accuracy : {accuracy_score(y_train, model.predict(X_train)):.3f}")
print(f"test accuracy  : {accuracy_score(y_test, model.predict(X_test)):.3f}")
print(f"test ROC AUC   : {roc_auc_score(y_test, model.predict(X_test)):.3f}")
