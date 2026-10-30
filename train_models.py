"""Trains both models once and saves the results to results.json for app.py.

The SVM takes over a minute per feature set, so the dashboard reads saved results.
Tree settings are the best ones found by the grid search in the notebook.
"""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from common import load_data, ORDER, CAT, SETS, TREE_PARAMS, cols_for

df = load_data()
X_all = pd.get_dummies(df[SETS['C: B + risk_score']], columns=CAT, dtype=int)
y = df.accident_severity
idx_tr, idx_te = train_test_split(df.index, test_size=.2, stratify=y, random_state=42)
y_tr, y_te = y.loc[idx_tr], y.loc[idx_te]


def score(yt, p):
    yt = np.asarray(yt); p = np.asarray(p); sv = yt != 'minor'
    r = recall_score(yt, p, labels=ORDER, average=None, zero_division=0)
    return {'Accuracy': accuracy_score(yt, p), 'Balanced accuracy': balanced_accuracy_score(yt, p),
            'Macro F1': f1_score(yt, p, average='macro', zero_division=0),
            'Minor recall': r[0], 'Major recall': r[1], 'Fatal recall': r[2],
            'Severe recall': (p[sv] != 'minor').mean(), 'Severe precision': precision_score(sv, p != 'minor', zero_division=0)}


out = {'metrics': [{'Feature set': 'Baseline', 'Model': 'Always "minor"', **score(y_te, np.full(len(y_te), 'minor'))}],
       'confusion': {}, 'importance': {}, 'by_type': []}
preds = {}
te = df.loc[idx_te].reset_index(drop=True)
for s, feats in SETS.items():
    cols = cols_for(X_all.columns, feats)
    Xtr, Xte = X_all.loc[idx_tr, cols], X_all.loc[idx_te, cols]
    tree = DecisionTreeClassifier(class_weight='balanced', random_state=42, **TREE_PARAMS[s]).fit(Xtr, y_tr)
    svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced', cache_size=300, random_state=42)).fit(Xtr, y_tr)
    preds[s] = {'Decision Tree': tree.predict(Xte), 'SVM': svm.predict(Xte)}
    for m, p in preds[s].items():
        out['metrics'].append({'Feature set': s, 'Model': m, **score(y_te, p)})
        out['confusion'][f'{s}|{m}'] = confusion_matrix(y_te, p, labels=ORDER).tolist()
    imp = pd.Series(tree.feature_importances_, index=cols)
    out['importance'][s] = {f: float(imp[cols_for(cols, [f])].sum()) for f in feats}

for g in ['accident_severity', 'weather', 'road_type', 'lighting', 'traffic_density', 'cause', 'visibility', 'vehicles_involved', 'city', 'festival']:
    for v, sub in te.groupby(g):
        row = {'Variable': g, 'Type': str(v), 'n': len(sub)}
        for s in SETS:
            for m in ('Decision Tree', 'SVM'):
                row[f'{s}|{m}'] = float((preds[s][m][sub.index] == sub.accident_severity.values).mean())
        out['by_type'].append(row)

with open('results.json', 'w') as f:
    json.dump(out, f, indent=1)
print(pd.DataFrame(out['metrics'])[['Feature set', 'Model', 'Accuracy', 'Fatal recall']].round(3))
