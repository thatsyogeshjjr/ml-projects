import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    'TV_Ad_Expenses': [i*100 for i in range(1, 11)],
    'Sales': [i for i in range(250, 2060, 200)]
}
print(data)
df = pd.DataFrame(data)

# Split data into features and target variables, x and y resp.
x = df['TV_Ad_Expenses'].values.reshape(-1, 1)
y = df['Sales'].values


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=.2, random_state=42)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


plt.scatter(x, y, color='blue', label='actual')
plt.scatter(x_test, y_pred, color='green', label='predicted (test)')
plt.plot(x_train, regressor.predict(x_train),
         color='red', label='predicted (train)')
plt.xlabel('tv advertisement expenses')
plt.ylabel('sales')
plt.title('linear regression -tv advertisement expenses vs sales')
plt.legend()
plt.show()
