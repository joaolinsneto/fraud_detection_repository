# Fraud Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Create the DataFrame directly with example data
data = {
    'transaction_id': [1, 2, 3, 4, 5],
    'user_id': [1001, 1002, 1001, 1003, 1001],
    'transaction_amount': [50.0, 200.0, 30.0, 500.0, 15.0],
    'transaction_time': ['2025-03-06 12:00:00', '2025-03-06 12:05:00', '2025-03-06 12:10:00', '2025-03-06 12:15:00', '2025-03-06 12:20:00'],
    'is_fraud': [0, 1, 0, 1, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# 2. View the first few records
print(df.head())

# 3. Analyze the data (simple example)
print(df.describe())

# 4. Select input variables (features) and output (target)
X = df[['transaction_amount']]  # Using only the transaction amount as a feature
y = df['is_fraud']  # The target variable (fraud or not)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Normalize the data (optional but recommended for ML algorithms)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train the model (Logistic Regression as an example)
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 10. Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. (Optional) Visualize the distribution of transactions
plt.figure(figsize=(8, 6))
plt.hist(df['transaction_amount'][df['is_fraud'] == 0], bins=30, alpha=0.5, label='No Fraud')
plt.hist(df['transaction_amount'][df['is_fraud'] == 1], bins=30, alpha=0.5, label='Fraud')
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.legend()
plt.show()