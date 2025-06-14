import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()

train = pd.read_csv('data/new_train_motion_data.csv', index_col=0)
test  = pd.read_csv('data/new_test_motion_data.csv', index_col=0)

X_train = train.drop('Class', axis=1)
X_train_scaled = scaler.fit_transform(X_train)

y_train = train['Class']

X_test  = test.drop('Class', axis=1)
X_test_scaled = scaler.transform(X_test)

y_test  = test['Class']

model = MLPClassifier(
    hidden_layer_sizes=(256, 64, 32),
    activation='relu',
    alpha=0.005,
    max_iter=500,
    early_stopping=True,
    learning_rate_init=0.05,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))

joblib.dump(model, 'xgb_simple_model.pkl')