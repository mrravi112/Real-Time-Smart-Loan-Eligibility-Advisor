import numpy as np
import pandas as pd
np.random.seed(42)
N = 3000

age = np.random.randint(22, 62, N)
monthly_income = np.random.randint(15000, 200000, N)
employment_type = np.random.choice(['Salaried', 'Self-Employed'], N, p=[0.65, 0.35])
credit_score = np.random.randint(300, 900, N)
loan_amount = np.random.randint(50000, 5000000, N)
loan_purpose = np.random.choice(['Home', 'Car', 'Personal', 'Business', 'Education'], N)
existing_emis = np.random.randint(0, 80000, N)
loan_tenure = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240], N)
dti = np.clip((existing_emis / np.maximum(monthly_income, 1)) * 100, 0, 100)
bank_behavior = np.clip((credit_score - 300) / 6 + np.random.normal(0, 8, N), 0, 100)

score = (
    (credit_score - 300) / 600 * 35 +
    np.clip(monthly_income / 200000, 0, 1) * 25 +
    (1 - np.clip(dti / 60, 0, 1)) * 20 +
    (employment_type == 'Salaried').astype(int) * 8 +
    bank_behavior / 100 * 8 +
    (1 - np.clip(loan_amount / (monthly_income * 60), 0, 1)) * 4
) + np.random.normal(0, 4, N)

eligible = (score > 50).astype(int)

df = pd.DataFrame({
    'age': age, 'monthly_income': monthly_income,
    'employment_type': employment_type, 'credit_score': credit_score,
    'loan_amount': loan_amount, 'loan_purpose': loan_purpose,
    'existing_emis': existing_emis, 'dti': dti.round(2),
    'loan_tenure': loan_tenure, 'bank_behavior': bank_behavior.round(1),
    'eligible': eligible
})
df.to_csv('/home/claude/smartloan/dataset/loan_data.csv', index=False)
print(f"Generated {N} samples | Approval rate: {eligible.mean()*100:.1f}%")
