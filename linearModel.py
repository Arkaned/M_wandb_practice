# @author: Arkaned
# date: 23/12/2024
# purpose: practice developing linear ML models and integrate with wandb

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load and prepare data
data = pd.DataFrame({
    'Size': [750, 800, 850, 900, 950],
    'Price': [150000, 160000, 165000, 180000, 190000]
})
X = data[['Size']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")