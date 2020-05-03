import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('flag.xlsx',header = 0)
print(data.head(8))
print(data.info())
print(data.columns)

X = data[["Zone","Area","Language","Religion","Red", "Green",
 "White","Circles","Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle","Animate",]]
y = data['Landmass']



score = []
for i in range(1,21):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.8)
    tree = DecisionTreeClassifier(max_depth=i)
    tree.fit(X_train,y_train)
    score.append(tree.score(X_test,y_test))

plt.scatter(range(1,21),score,alpha = 0.6,color='r')
plt.show()