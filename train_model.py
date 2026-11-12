"""Train an INDEPENDENT Python RF for standalone demonstration, not SAS Viya's model."""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from features import prepare, FEATURES

ROOT = Path(__file__).resolve().parent
INPUT = ROOT/'data'/'Steel_industry_data.csv'
MODELS = ROOT/'models'
MODELS.mkdir(exist_ok=True)
raw = pd.read_csv(INPUT, encoding='utf-8-sig')
d = prepare(raw)
train = d.iloc[:24057].copy()
test = d.iloc[24057:].copy()
assert len(train) == 24057 and len(test) == 10311
assert train.timestamp.max() < test.timestamp.min()
assert not d[FEATURES + ['Load_Type']].isna().any().any()
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=12345,
    n_jobs=-1,
)
model.fit(train[FEATURES], train['Load_Type'])
pred = model.predict(test[FEATURES])
classes = model.classes_.tolist()
metrics = {
    'model_origin': 'Independent Python scikit-learn RandomForestClassifier (NOT SAS Viya forest)',
    'model_parameters': {'n_estimators': 100, 'max_depth': 15, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 12345},
    'train_rows': len(train), 'test_rows': len(test),
    'train_from': str(train.timestamp.min()), 'train_through': str(train.timestamp.max()),
    'test_from': str(test.timestamp.min()), 'test_through': str(test.timestamp.max()),
    'classes': classes,
    'accuracy_test': float(accuracy_score(test['Load_Type'], pred)),
    'macro_f1_test': float(f1_score(test['Load_Type'], pred, average='macro')),
    'weighted_f1_test': float(f1_score(test['Load_Type'], pred, average='weighted')),
    'confusion_matrix': confusion_matrix(test['Load_Type'], pred, labels=classes).tolist(),
    'classification_report': classification_report(test['Load_Type'], pred, output_dict=True, zero_division=0),
}
joblib.dump({'model': model, 'features': FEATURES, 'classes': classes}, MODELS/'python_random_forest.joblib')
(MODELS/'python_validation_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
example_cols = ['timestamp', 'Usage_kWh', 'Load_Type'] + FEATURES
examples = test.iloc[[0, min(150,len(test)-1), min(800,len(test)-1), min(2500,len(test)-1)]][example_cols]
examples.to_csv(ROOT/'data'/'demo_examples.csv', index=False)
print('Training done. Python Random Forest is independent of SAS Viya.')
print('Rows train/test:', len(train), len(test))
print('Test accuracy:', round(metrics['accuracy_test'],4), 'Macro F1:', round(metrics['macro_f1_test'],4))
print('Saved models/python_random_forest.joblib and models/python_validation_metrics.json')
