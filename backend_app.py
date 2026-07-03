from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json, numpy as np, os

app = Flask(__name__)
CORS(app)

MD = '/home/claude/smartloan/models'
rf  = joblib.load(f'{MD}/random_forest.pkl')
gb  = joblib.load(f'{MD}/gradient_boosting.pkl')
nn  = joblib.load(f'{MD}/neural_network.pkl')
dt  = joblib.load(f'{MD}/decision_tree.pkl')
lr  = joblib.load(f'{MD}/logistic_regression.pkl')
ens = joblib.load(f'{MD}/ensemble.pkl')
scaler = joblib.load(f'{MD}/scaler.pkl')
le_emp = joblib.load(f'{MD}/le_emp.pkl')
le_pur = joblib.load(f'{MD}/le_purpose.pkl')

with open(f'{MD}/results.json') as f:
    analytics = json.load(f)

EMP_MAP = {'Salaried': 0, 'Self-Employed': 1}
PUR_MAP = {'Business': 0, 'Car': 1, 'Education': 2, 'Home': 3, 'Personal': 4}
FEAT_NAMES = ['Age','Monthly Income','Employment','Credit Score','Loan Amount',
              'Loan Purpose','Existing EMIs','DTI Ratio','Tenure','Bank Behavior']

def build_features(d):
    emp = EMP_MAP.get(d['employment_type'], 0)
    pur = PUR_MAP.get(d['loan_purpose'], 4)
    dti = round((d['existing_emis'] / max(d['monthly_income'], 1)) * 100, 2)
    bank_beh = min(100, max(0, (d['credit_score'] - 300) / 6))
    return np.array([[d['age'], d['monthly_income'], emp, d['credit_score'],
                      d['loan_amount'], pur, d['existing_emis'],
                      dti, d['loan_tenure'], bank_beh]]), dti, bank_beh

def risk_score(prob):
    return round((1 - prob) * 100, 1)

def suggest_rate(prob, credit):
    if prob >= 0.80: base = 7.5
    elif prob >= 0.65: base = 10.0
    elif prob >= 0.50: base = 13.5
    else: base = 17.0
    adj = max(0, (700 - credit) / 100) * 0.5
    return round(base + adj, 2)

def calc_emi(amount, rate_annual, months):
    r = rate_annual / 100 / 12
    if r == 0: return round(amount / months, 0)
    return round(amount * r * (1+r)**months / ((1+r)**months - 1), 0)

def generate_explanations(d, prob, dti):
    reasons, tips = [], []
    if d['credit_score'] < 600:
        reasons.append(f"Low credit score ({d['credit_score']}) — minimum 650 recommended")
        tips.append({'icon':'📊','msg':f"Raise credit score to 700+ by clearing dues. Current: {d['credit_score']}→700 could add ~15% approval chance.",'delta':15})
    if dti > 40:
        reasons.append(f"High debt-to-income ratio ({dti:.1f}%) — above 40% threshold")
        tips.append({'icon':'💳','msg':f"Reduce existing EMIs by ₹{int(d['existing_emis']*0.3):,}/mo to bring DTI under 40%.",'delta':12})
    if d['monthly_income'] < 30000:
        reasons.append(f"Monthly income (₹{d['monthly_income']:,}) below preferred minimum of ₹30,000")
        tips.append({'icon':'💰','msg':"Increasing income to ₹50,000/mo could improve approval chance by ~20%.",'delta':20})
    lti = d['loan_amount'] / max(d['monthly_income'] * 12, 1)
    if lti > 6:
        reasons.append(f"Loan amount is {lti:.1f}× annual income — consider requesting less")
        tips.append({'icon':'📉','msg':f"Reducing loan by ₹{int(d['loan_amount']*0.25):,} could help approval significantly.",'delta':10})
    if d['employment_type'] == 'Self-Employed':
        tips.append({'icon':'💼','msg':"Salaried applicants have ~8% higher baseline approval rates.",'delta':8})
    if not reasons:
        reasons.append("Good overall profile — minor optimizations possible")
    if not tips:
        tips.append({'icon':'✅','msg':"Your profile looks strong! Consider increasing tenure for lower EMIs.",'delta':0})
    return reasons[:3], tips[:4]

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    X, dti, bank_beh = build_features(d)
    Xsc = scaler.transform(X)

    probs = {
        'Random Forest':       float(rf.predict_proba(X)[0][1]),
        'Gradient Boosting':   float(gb.predict_proba(X)[0][1]),
        'Neural Network':      float(nn.predict_proba(Xsc)[0][1]),
        'Decision Tree':       float(dt.predict_proba(X)[0][1]),
        'Logistic Regression': float(lr.predict_proba(Xsc)[0][1]),
    }
    weights = [0.30, 0.30, 0.20, 0.10, 0.10]
    prob = sum(p*w for p, w in zip(probs.values(), weights))
    eligible = prob >= 0.50
    rate = suggest_rate(prob, d['credit_score'])
    emi  = calc_emi(d['loan_amount'], rate, d['loan_tenure'])
    reasons, tips = generate_explanations(d, prob, dti)

    return jsonify({
        'eligible': eligible,
        'verdict': 'Approved' if eligible else 'Rejected',
        'probability': round(prob * 100, 2),
        'risk_score': risk_score(prob),
        'dti': round(dti, 2),
        'bank_behavior': round(bank_beh, 1),
        'suggested_rate': rate,
        'estimated_emi': int(emi),
        'model_probs': {k: round(v*100,2) for k,v in probs.items()},
        'reasons': reasons,
        'tips': tips,
        'feature_importances': analytics['feature_importances'],
    })

@app.route('/simulate', methods=['POST'])
def simulate():
    """What-if simulator — batch predictions for slider changes"""
    base = request.json.get('base', {})
    changes = request.json.get('changes', {})
    scenarios = []
    for label, delta in changes.items():
        d2 = {**base}
        if label == 'income':   d2['monthly_income'] = max(1, base['monthly_income'] + delta)
        elif label == 'credit': d2['credit_score']   = max(300, min(900, base['credit_score'] + delta))
        elif label == 'emis':   d2['existing_emis']  = max(0, base['existing_emis'] + delta)
        elif label == 'amount': d2['loan_amount']     = max(10000, base['loan_amount'] + delta)
        X2, _, _ = build_features(d2)
        Xsc2 = scaler.transform(X2)
        weights = [0.30, 0.30, 0.20, 0.10, 0.10]
        probs2 = [float(rf.predict_proba(X2)[0][1]), float(gb.predict_proba(X2)[0][1]),
                  float(nn.predict_proba(Xsc2)[0][1]), float(dt.predict_proba(X2)[0][1]),
                  float(lr.predict_proba(Xsc2)[0][1])]
        p2 = sum(x*w for x, w in zip(probs2, weights))
        scenarios.append({'label': label, 'delta': delta, 'probability': round(p2*100,2), 'eligible': p2>=0.5})
    return jsonify({'scenarios': scenarios})

@app.route('/analytics', methods=['GET'])
def get_analytics():
    return jsonify(analytics)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
