import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

np.random.seed(101)
n = 4000

# 1. Generate Telecom Customer Churn Dataset
df = pd.DataFrame({
    'Customer_ID': [f'CUST{2000+i}' for i in range(n)],
    'Tenure_Months': np.random.randint(1, 72, n),
    'Monthly_Charges_INR': np.random.normal(799, 250, n).clip(299, 1999).round(0),
    'Contract_Type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n, p=[0.55, 0.25, 0.20]),
    'Support_Tickets': np.random.poisson(1.8, n),
    'Experiment_Group': np.random.choice(['Control_No_Discount', 'Treatment_15Pct_Loyalty'], n)
})

logit = (
    -1.2
    - 0.035 * df['Tenure_Months']
    + 0.0015 * df['Monthly_Charges_INR']
    + 0.35 * df['Support_Tickets']
    + np.where(df['Contract_Type'] == 'Month-to-Month', 1.1, 0)
    + np.where(df['Experiment_Group'] == 'Treatment_15Pct_Loyalty', -0.45, 0)
)
df['Churn'] = np.random.binomial(1, 1 / (1 + np.exp(-logit)))
df.to_csv('telecom_customer_churn_ab_data.csv', index=False)

# 2. A/B Test (Two-Proportion Z-Test)
control = df[df['Experiment_Group'] == 'Control_No_Discount']['Churn']
treatment = df[df['Experiment_Group'] == 'Treatment_15Pct_Loyalty']['Churn']
p1, p2 = control.mean(), treatment.mean()
p_pool = (control.sum() + treatment.sum()) / (len(control) + len(treatment))
z_stat = (p1 - p2) / np.sqrt(p_pool * (1 - p_pool) * (1/len(control) + 1/len(treatment)))
p_val_ab = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"Control Churn: {p1*100:.2f}% | Treatment Churn: {p2*100:.2f}% | Z = {z_stat:.3f}, p = {p_val_ab:.4e}")

# 3. Logistic Regression
model = smf.logit('Churn ~ Tenure_Months + Monthly_Charges_INR + Support_Tickets + C(Contract_Type)', data=df).fit(disp=False)
print("\n=== ODDS RATIOS ===")
print(np.exp(model.params).round(4))

# 4. Generate Clean 2-Panel Visual Report
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#F8F9FA')
fig.suptitle('TELECOM CHURN & A/B LOYALTY EXPERIMENT ANALYSIS', fontsize=14, fontweight='bold', color='#1B365D')

# Panel 1: A/B Test Results
ab_rates = [p1*100, p2*100]
bars = axes[0].bar(['Control\n(No Discount)', 'Treatment\n(15% Loyalty Offer)'], ab_rates, color=['#D9534F', '#2E8B57'], width=0.5)
axes[0].set_title(f'A/B Experiment Churn Rate (Z = {z_stat:.2f}, p < 0.001)', fontweight='bold', color='#1B365D')
axes[0].set_ylabel('Churn Rate (%)')
axes[0].set_ylim(0, max(ab_rates)*1.25)
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{bar.get_height():.1f}%", ha='center', fontweight='bold')

# Panel 2: Churn by Contract Type
contract_churn = df.groupby('Contract_Type')['Churn'].mean().sort_values(ascending=False) * 100
sns.barplot(x=contract_churn.index, y=contract_churn.values, palette=['#D9534F', '#F0AD4E', '#0066CC'], ax=axes[1])
axes[1].set_title('Churn Rate (%) by Contract Type', fontweight='bold', color='#1B365D')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_xlabel('')
axes[1].set_ylim(0, max(contract_churn.values)*1.25)
for i, v in enumerate(contract_churn.values):
    axes[1].text(i, v + 0.8, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('churn_ab_analysis.png', dpi=300)
plt.show()



