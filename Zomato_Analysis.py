#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install faker')



# In[3]:


import pandas as pd
import numpy as np
import random
from faker import Faker

# 1. Setup
fake = Faker()
np.random.seed(42)
num_users = 100000  # <--- Change this to 1 Million if you want huge data!

# 2. Generate Base Data
data = {
    'user_id': [f'U{i}' for i in range(1000, 1000 + num_users)],
    'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Chennai'], num_users, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
    'membership': np.random.choice(['Gold', 'Regular'], num_users, p=[0.4, 0.6]),
    'join_date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_users)]
}

df = pd.DataFrame(data)

# 3. Add Behavioral Metrics (The Logic)
# Gold members order more frequently
df['avg_monthly_orders'] = np.where(df['membership']=='Gold', 
                                    np.random.normal(8, 2, num_users), 
                                    np.random.normal(3, 1, num_users))
df['avg_monthly_orders'] = df['avg_monthly_orders'].clip(lower=1).astype(int)

# Rain Fees (Higher in Mumbai/Bangalore)
rain_prob = df['city'].map({'Mumbai': 0.8, 'Bangalore': 0.6, 'Delhi': 0.2, 'Pune': 0.3, 'Chennai': 0.4})
df['rain_fees_paid'] = (np.random.binomial(n=5, p=rain_prob, size=num_users) * 25) # 25 Rs per fee

# Delivery Issues (Proxy for bad experience)
df['late_deliveries_last_month'] = np.random.poisson(lam=0.5, size=num_users)

# Average Order Value (AOV)
df['aov'] = np.random.normal(450, 150, num_users).astype(int)

# 4. Define Churn Logic (Ground Truth)
# IF (Rain Fees > 50 AND Membership = Gold) OR (Late Deliveries > 2) -> High Risk of Churn
conditions = [
    (df['membership'] == 'Gold') & (df['rain_fees_paid'] > 50),
    (df['late_deliveries_last_month'] > 2),
    (df['avg_monthly_orders'] < 2)
]
choices = [0.65, 0.80, 0.50] # Probability of Churn for each condition
df['churn_prob'] = np.select(conditions, choices, default=0.10) # Base churn is 10%

# Add some randomness to churn so it's not perfect
df['churn_status'] = np.random.binomial(1, df['churn_prob'])

# 5. Export
print(f"Generated {len(df)} rows.")
print(df.head())
df.to_csv('zomato_gold_churn_dataset.csv', index=False)


# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('zomato_gold_churn_dataset.csv')

# Filtering for GOLD Members only (The population of interest)
gold_users = df[df['membership'] == 'Gold'].copy()

# Create Segments based on Rain Fees Paid
# Segment A: "Happy Users" (Paid 0 Fees)
# Segment B: "Annoyed Users" (Paid small fees, < 50)
# Segment C: "Angry Users" (Paid high fees, > 50)
def categorize_fees(amount):
    if amount == 0:
        return '0_No_Fee'
    elif amount <= 50:
        return '1_Low_Fee'
    else:
        return '2_High_Fee'

gold_users['fee_segment'] = gold_users['rain_fees_paid'].apply(categorize_fees)

#Calculate Churn Rate per Segment
churn_analysis = gold_users.groupby('fee_segment').agg({
    'user_id': 'count',       
    'churn_status': 'mean',      
    'aov': 'mean'              
}).rename(columns={'user_id':'Total_Users', 'churn_status':'Churn_Rate'})

print(churn_analysis)

# Calculate Financial Impact (The "Steamroller" Math)
# Avg Annual Spend = Monthly Orders * AOV * 12
avg_annual_spend = gold_users['avg_monthly_orders'].mean() * gold_users['aov'].mean() * 12

print(f"Average Annual Revenue per Gold User: ₹{avg_annual_spend:,.2f}")


# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


try:
    df = pd.read_csv('zomato_gold_churn_dataset.csv')
except FileNotFoundError:
    print("Error: csv file not found. Please generate the data using Option 1 script first.")
    exit()

# Filter for GOLD members only (since they are the focus of our problem)
gold_df = df[df['membership'] == 'Gold'].copy()

# Create 'Whale' Segments based on Spend
# Logic: If they order > 5 times a month OR spend > 800 per order, they are High Value
gold_df['customer_segment'] = np.where(
    (gold_df['avg_monthly_orders'] * gold_df['aov']) > 3000, 
    'High Value (Whale)', 
    'Low Value (Casual)'
)


sns.set_theme(style="whitegrid")

# CHART 1: THE CONTEXT (Membership Churn) ---
# Question: Do Gold members churn more than Regular members?
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='membership', y='churn_status', palette=['#FFD700', '#C0C0C0']) # Gold & Silver colors
plt.title('Churn Rate: Gold Members vs. Regular Users', fontsize=14, fontweight='bold')
plt.ylabel('Churn Rate (Probability)', fontsize=12)
plt.xlabel('Membership Type', fontsize=12)
# Add average line
plt.axhline(df['churn_status'].mean(), color='red', linestyle='--', label='Global Average')
plt.legend()
plt.tight_layout()
plt.show()



# In[11]:


# --- CHART 2: THE SMOKING GUN (The "Steamroller" Chart) ---
# Question: As Rain Fees increase, does Churn increase?
plt.figure(figsize=(10, 6))
# Calculate Churn Rate per Fee Bucket
fee_churn = gold_df.groupby('rain_fees_paid')['churn_status'].mean().reset_index()

# Plot Line Chart
sns.lineplot(data=fee_churn, x='rain_fees_paid', y='churn_status', marker='o', color='#cb202d', linewidth=2.5)
plt.title('Impact of Rain Fees on Gold Member Churn', fontsize=14, fontweight='bold')
plt.ylabel('Churn Rate (%)', fontsize=12)
plt.xlabel('Total Rain Fees Paid (₹)', fontsize=12)

# Annotate the "Inflection Point"
plt.axvline(x=75, color='grey', linestyle='--')
plt.text(80, 0.4, '⚠️ Inflection Point:\nChurn spikes after ₹75', fontsize=10, color='red')
plt.tight_layout()
plt.show()


# In[12]:


# --- CHART 3: THE VICTIM (Whale vs. Casual Sensitivity) ---
# Question: Who are we angering? The rich or the poor?
plt.figure(figsize=(10, 6))
sns.barplot(data=gold_df, x='customer_segment', y='churn_status', hue='rain_fees_paid', palette='Reds')
plt.title('Price Sensitivity: "Whales" vs. "Casuals"', fontsize=14, fontweight='bold')
plt.ylabel('Churn Rate', fontsize=12)
plt.xlabel('Customer Value Segment', fontsize=12)
plt.legend(title='Rain Fees Paid (₹)')
plt.tight_layout()
plt.show()


# In[13]:


# --- CHART 4: THE VERDICT (Financial Impact) ---
# Question: Are we making money or losing money?
# Calculation
total_fees_collected = gold_df['rain_fees_paid'].sum()
# Loss = Churned Users * Their Annual Value
churned_whales = gold_df[(gold_df['customer_segment'] == 'High Value (Whale)') & (gold_df['churn_status'] == 1)]
revenue_lost = (churned_whales['avg_monthly_orders'] * churned_whales['aov'] * 12).sum()

# Prepare Data for Chart
financials = pd.DataFrame({
    'Metric': ['Rain Fees Collected', 'Annual Revenue Lost'],
    'Amount': [total_fees_collected, revenue_lost]
})

plt.figure(figsize=(8, 6))
ax = sns.barplot(data=financials, x='Metric', y='Amount', palette=['green', 'red'])
plt.title('Financial Disaster: Fees Gained vs. Value Lost', fontsize=14, fontweight='bold')
plt.ylabel('Amount (INR)', fontsize=12)

# Add text labels on bars
for p in ax.patches:
    ax.annotate(f'₹{p.get_height():,.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.show()


# In[ ]:




