import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
import pickle
from collections import Counter
import seaborn as sns
# Load data
data = pd.read_csv("traffic volume.csv")

# Handle missing values
data['temp'].fillna(data['temp'].mean(), inplace=True)
data['rain'].fillna(data['rain'].mean(), inplace=True)
data['snow'].fillna(data['snow'].mean(), inplace=True)
data['weather'] = data['weather'].fillna('Clouds')

# Split date and time
data[["day", "month", "year"]] = data["date"].str.split("-", expand=True)
data[["hours", "minutes", "seconds"]] = data["Time"].str.split(":", expand=True)
data.drop(columns=['date', 'Time'], axis=1, inplace=True)


corr = data.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")

sns.pairplot(data)
data.boxplot()
# Prepare features
y = data['traffic_volume']
x = data.drop(columns=['traffic_volume'], axis=1)

# Encode categorical columns
cat_cols = ['holiday', 'weather']
le = LabelEncoder()
for col in cat_cols:
    x[col] = le.fit_transform(x[col])

# Save encoder only if needed in future
# pickle.dump(le, open("encoder.pkl", "wb"))

# Scale features using StandardScaler (IMPORTANT)
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train model
model = ensemble.RandomForestRegressor()
model.fit(x_train, y_train)

# Save model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully.")