import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#opening the E-Commerce csv File
customer = pd.read_csv("Ecommerce Customers")

print(customer.head(), "\n", customer.describe(), "\n", customer.info())

sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

sns.jointplot(x = "Time on Website", y = "Yearly Amount Spent", data = customer)

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customer)

sns.jointplot(x='Time on App',y='Length of Membership',data=customer, kind = 'hex')

sns.pairplot(customer)

#Question, which plot looks to be the most correlated feature of Yearly Amout Spent ?
#Ans = Length of Membership

sns.lmplot(y = 'Yearly Amount Spent', x = 'Length of Membership', data = customer)


#To show all the plots
plt.show()

X = customer[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = customer['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()


lm.fit(X_train,y_train)

print('Coefficients: \n', lm.coef_)


predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

plt.show()

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

sns.distplot((y_test-predictions),bins=50);
plt.show()


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)

