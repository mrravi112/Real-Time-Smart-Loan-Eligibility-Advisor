import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score, precision_score, recall_score)
from sklearn.neural_network import MLPClassifier

MODEL_DIR = '/home/claude/smartloan/models'
os.makedirs(MODEL_DIR, exist_ok=True)

def get_feature_names():
    return ['age', 'monthly_income', 'employment_encoded', 'credit_score',
            'loan_amount', 'purpose_encoded', 'existing_emis', 'dti', 'loan_tenure', 'bank_behavior']

def train_all():
    df = pd.read_csv('/home/claude/smartloan/dataset/loan_data.csv')

    le_emp = LabelEncoder()
    le_purpose = LabelEncoder()
    df['employment_encoded'] = le_emp.fit_transform(df['employment_type'])
    df['purpose_encoded'] = le_purpose.fit_transform(df['loan_purpose'])

    features = get_feature_names()
    X = df[features]
    y = df['eligible']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models_def = {
        'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000, C=0.5), True),
        'Decision Tree':       (DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_leaf=10), False),
        'Random Forest':       (RandomForestClassifier(random_state=42, n_estimators=150, max_depth=12), False),
        'Gradient Boosting':   (GradientBoostingClassifier(random_state=42, n_estimators=150, max_depth=5, learning_rate=0.08), False),
        'Neural Network':      (MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42, max_iter=300, early_stopping=True, learning_rate_init=0.001), True),
    }

    results = {}
    trained_models = {}
    roc_data = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (model, needs_scale) in models_def.items():
        Xtr = X_train_sc if needs_scale else X_train
        Xte = X_test_sc if needs_scale else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        cm   = confusion_matrix(y_test, y_pred).tolist()

        cv_data = X_train_sc if needs_scale else X_train
        cv = cross_val_score(model, cv_data, y_train, cv=skf, scoring='accuracy')

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {'fpr': fpr.tolist()[::5], 'tpr': tpr.tolist()[::5]}

        results[name] = {
            'accuracy': round(acc * 100, 2), 'auc': round(auc, 4),
            'f1': round(f1 * 100, 2), 'precision': round(prec * 100, 2), 'recall': round(rec * 100, 2),
            'cv_mean': round(cv.mean() * 100, 2), 'cv_std': round(cv.std() * 100, 2),
            'confusion_matrix': cm
        }
        trained_models[name] = model
        print(f"✓ {name:25s} Acc={acc*100:.2f}% AUC={auc:.4f} CV={cv.mean()*100:.2f}±{cv.std()*100:.2f}%")

    # Ensemble (Voting)
    estimators = [
        ('rf',  trained_models['Random Forest']),
        ('gb',  trained_models['Gradient Boosting']),
        ('dt',  trained_models['Decision Tree']),
    ]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    ensemble.fit(X_train, y_train)
    ens_pred = ensemble.predict(X_test)
    ens_prob = ensemble.predict_proba(X_test)[:, 1]
    ens_acc  = accuracy_score(y_test, ens_pred)
    ens_auc  = roc_auc_score(y_test, ens_prob)
    results['Ensemble (Voting)'] = {
        'accuracy': round(ens_acc*100,2), 'auc': round(ens_auc,4),
        'f1': round(f1_score(y_test,ens_pred)*100,2),
        'precision': round(precision_score(y_test,ens_pred)*100,2),
        'recall': round(recall_score(y_test,ens_pred)*100,2),
        'cv_mean': round(ens_acc*100,2), 'cv_std': 0,
        'confusion_matrix': confusion_matrix(y_test,ens_pred).tolist()
    }
    fpr, tpr, _ = roc_curve(y_test, ens_prob)
    roc_data['Ensemble (Voting)'] = {'fpr': fpr.tolist()[::5], 'tpr': tpr.tolist()[::5]}
    trained_models['Ensemble'] = ensemble
    print(f"✓ {'Ensemble (Voting)':25s} Acc={ens_acc*100:.2f}% AUC={ens_auc:.4f}")

    # Feature importances (RF)
    rf = trained_models['Random Forest']
    feat_names = ['Age', 'Monthly Income', 'Employment', 'Credit Score', 'Loan Amount',
                  'Loan Purpose', 'Existing EMIs', 'DTI Ratio', 'Tenure', 'Bank Behavior']
    importances = {n: round(float(v)*100, 2) for n, v in zip(feat_names, rf.feature_importances_)}

    # Save
    joblib.dump(trained_models['Random Forest'], f'{MODEL_DIR}/random_forest.pkl')
    joblib.dump(trained_models['Gradient Boosting'], f'{MODEL_DIR}/gradient_boosting.pkl')
    joblib.dump(trained_models['Neural Network'], f'{MODEL_DIR}/neural_network.pkl')
    joblib.dump(trained_models['Decision Tree'], f'{MODEL_DIR}/decision_tree.pkl')
    joblib.dump(trained_models['Logistic Regression'], f'{MODEL_DIR}/logistic_regression.pkl')
    joblib.dump(trained_models['Ensemble'], f'{MODEL_DIR}/ensemble.pkl')
    joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')
    joblib.dump(le_emp, f'{MODEL_DIR}/le_emp.pkl')
    joblib.dump(le_purpose, f'{MODEL_DIR}/le_purpose.pkl')

    payload = {'model_results': results, 'feature_importances': importances, 'roc_data': roc_data}
    with open(f'{MODEL_DIR}/results.json', 'w') as f:
        json.dump(payload, f, indent=2)

    print("\nAll models saved ✓")
    return results, importances

if __name__ == '__main__':
    train_all()
