
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
np.random.seed(42)
x = 2.5 * np.random.randn(100) + 1.5
y = 0.5 * x + np.random.randn(100) * 0.2 + 0.2
data = pd.DataFrame({'Height': x, 'Weight': y})
data.to_csv('synthetic-data.csv', index=False)
df = pd.read_csv('synthetic-data.csv')
x = df[['Height']]
y = df[['Weight']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.3f}")
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()
