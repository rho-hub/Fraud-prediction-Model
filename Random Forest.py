#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\USER\Desktop\fraud detection pipeline\creditcard.csv")

# View first few rows
df.head(10)


# In[2]:


from imblearn.over_sampling import SMOTE
# Split data first
from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE only to training data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print('Original dataset shape:', y_train.value_counts())
print('Resampled dataset shape:', y_train_res.value_counts())


# In[3]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train on the resampled data
rf_model.fit(X_train_res, y_train_res)

# Predict on real test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[4]:


# Feature importance
importances = rf_model.feature_importances_
features = X.columns

# Create a dataframe
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display
print(feature_importance_df.head(10))  # Top 10 important features

# Plot
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Important Features')
plt.show()


# In[ ]:




