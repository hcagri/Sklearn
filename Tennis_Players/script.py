import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# load the data and just analyze the variables
df = pd.read_csv("tennis_stats.csv")
print(df.info())
print(df.head())
# outcomes are Wins, Losses, Winnings, Raking


# perform exploratory analysis

plt.scatter(df["BreakPointsOpportunities"],df["Winnings"],alpha=0.4)
plt.xlabel("Break Points Opportunities")
plt.ylabel("Winnings")
plt.title("Analysis 1 ")
plt.show()

plt.scatter(df["DoubleFaults"],df["Losses"],alpha=0.4)
plt.xlabel("Double Faults")
plt.ylabel("Losses")
plt.title("Analysis 2 ")
plt.show()


# perform single feature linear regressions 
X = df["DoubleFaults"]
X = X.values.reshape(-1,1)
y = df["Losses"]
y = y.values.reshape(-1,1)

single_reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.8,train_size = 0.2,random_state = 42)
single_reg.fit(X_train,y_train)
y_prediction = single_reg.predict(X_test)

plt.scatter(y_test,y_prediction,alpha = 0.4,marker = "x")
plt.xlabel("Actual Losses Values")
plt.ylabel("Predicted Losses Values")
plt.title("Linear Regression with 1 variable")
plt.show()

y_prediction = single_reg.predict(X_train) 
plt.plot(X_train,y_prediction,'r')
plt.title("Linear Regression with 1 variable Prediction Line")
plt.show()

print('Predicting Losses with DoubleFaults Test Score:', single_reg.score(X_test,y_test))

## perform two feature linear regression


X = df[["BreakPointsOpportunities", "DoubleFaults"]]
y = df["Winnings"]
y = y.values.reshape(-1,1)

two_reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.8,train_size = 0.2,random_state = 42)
two_reg.fit(X_train,y_train)
y_prediction = two_reg.predict(X_test)

plt.scatter(y_test,y_prediction,alpha = 0.4,marker = 'x')
plt.xlabel("Actual Winnings Values")
plt.ylabel("Predicted Winnings Values")
plt.title("Linear Regression with 2 variable")
plt.show()
print('Predicting Winnings with BreakPointsOpportunities and DoubleFaults Test Score:', two_reg.score(X_test,y_test))

## perform multiple feature linear regressions


X = df.iloc[:,2:20]
y = df["Winnings"]
y = y.values.reshape(-1,1)

multi_reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.8,train_size = 0.2,random_state = 42)
multi_reg.fit(X_train,y_train)
y_prediction = multi_reg.predict(X_test)

plt.scatter(y_test,y_prediction,alpha = 0.4,marker = 'x')
plt.xlabel("Actual Winnings Values")
plt.ylabel("Predicted Winnings Values")
plt.title("The predictions vs actual test values for multiple linear Regression Winnings")
plt.show()
print('Predicting Winnings with Multi Features Test Score:', multi_reg.score(X_test,y_test))

## perform multiple feature linear regressions


X = df.iloc[:,2:20]
y = df["Losses"]
y = y.values.reshape(-1,1)

multi_reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.8,train_size = 0.2,random_state = 42)
multi_reg.fit(X_train,y_train)
y_prediction = multi_reg.predict(X_test)

plt.scatter(y_test,y_prediction,alpha = 0.4)
plt.xlabel("Actual Winnings Values")
plt.ylabel("Predicted Winnings Values")
plt.title("The predictions vs actual test values for multiple linear Regression Losses")
plt.show()
print('Predicting Losses with Multi Features Test Score:', multi_reg.score(X_test,y_test))

