import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree



data = pd.read_csv('income.csv',header = 0,engine='python',delimiter=", ")

data['age'] = data['"age'].apply(lambda x: x[1:])
data['income'] = data['income"'].apply(lambda x: x[:-1])
data['sex_int'] = data['sex'].apply(lambda x: 0 if x =='Male' else 1)
data['country_int'] = data['native-country'].apply(lambda x: 0 if x =='United-States' else 1)
data.drop(['income"','"age'],axis = 1,inplace=True)

print(data.head(10))
print(data.info())



# random forest can't use String variables. They must be integer or float.
X = data[['fnlwgt','education-num','age','capital-gain','capital-loss','hours-per-week','sex_int','country_int']]
labels = data['income']

X_train, X_test, y_train, y_test = train_test_split(X,labels,random_state=42, test_size=0.8)

model = RandomForestClassifier(random_state=0)
model.fit(X_train,y_train)
print(list(zip(X.columns,model.feature_importances_)))

print('Model Score is: %'+str(100*model.score(X_test,y_test)))
