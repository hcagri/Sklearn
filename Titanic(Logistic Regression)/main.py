import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print()

# ============ İmport Data ===========
data = pd.read_csv('train.csv')

survived = data["Survived"]

# ================ Analyze and Visualize the Data ==================
class_1_survived = 0
class_1_died = 0
class_2_survived = 0
class_2_died = 0
class_3_survived = 0
class_3_died = 0

for i in range(len(survived)):
    if data["Pclass"][i] == 1:
        if survived[i] == 1 :
            class_1_survived += 1
        else:
            class_1_died += 1
    
    elif data["Pclass"][i] == 2:
        if survived[i] == 1 :
            class_2_survived += 1
        else:
            class_2_died += 1

    elif data["Pclass"][i] == 3:
        if survived[i] == 1 :
            class_3_survived += 1
        else:
            class_3_died += 1

fig, axs = plt.subplots(1,3)
fig.suptitle("Survivals of Passangers for each class")
axs[0].bar([0 , 1 ],[class_1_died, class_1_survived],tick_label=["Died","Survived"],color = ['r','b'],alpha=0.4)
axs[0].set_title("Class 1")
axs[1].bar([0 , 1 ],[class_2_died, class_2_survived],tick_label=["Died","Survived"],color = ['r','b'],alpha=0.4)
axs[1].set_title("Class 2")
axs[2].bar([0 , 1 ],[class_3_died, class_3_survived],tick_label=["Died","Survived"],color = ['r','b'],alpha=0.4)
axs[2].set_title("Class 3")
plt.show()

survived_with_cabin = 0
die_with_cabin = 0
bool_series_cabin = data["Cabin"].isnull()
for i in range(len(data["Cabin"])):
    if bool_series_cabin[i]:
        continue
    else:
        if survived[i] == 1:
            survived_with_cabin +=1
        else:
            die_with_cabin += 1


print("# of survivals who has Cabin : {} ".format(survived_with_cabin))
print("# of total survivals : {} ".format(class_1_survived + class_2_survived + class_3_survived))




# =========== Clean the Data ================


# İf passanger has a cabin represent it with 1. Otherwise 0
X_features = pd.DataFrame()
X_features["Cabin"] = data["Cabin"].isnull().apply(lambda x: 0 if x else 1)

# if passanger is female = 1, male = 0. Because first womans and childrens are rescued.
X_features["Sex"] = data["Sex"].apply(lambda x: 1 if x =='female' else 0)
    
# As can seen in the analyze section The person who is a first class passanger 
# has a higher change to survive

X_features["First_class"] = data["Pclass"].apply(lambda x: 1 if x == 1 else 0)
X_features["Second_class"] = data["Pclass"].apply(lambda x: 1 if x == 2 else 0)

# The age is important but it has null values, we should replace these with the mean.
X_features["Age"] = data["Age"]
mean_age = X_features["Age"].mean()
X_features["Age"].fillna(mean_age, inplace=True)




# normalize the feature matrix
scale = StandardScaler()
X_feature_normalized = scale.fit_transform(X_features)


# First split the data to train and test set in order to estimate the score
X_train, X_test, y_train, y_test = train_test_split(X_feature_normalized, survived, test_size=0.9, random_state=42)

# Create a model for logistic regression and test
model = LogisticRegression()
model.fit(X_train,y_train)
print()
print("The score of the model on validation set. {} ".format(model.score(X_test,y_test)))
print()
print("The weights of each Feature :")
print(list(zip(['Cabin','Sex','First_class','Second_class','Age'],model.coef_[0])))


# ======== Make a prediction for unseen events. This part for Kaggle Competition ============

test_data = pd.read_csv('test.csv')

# Clean the test data

Test_data_clean = pd.DataFrame()
Test_data_clean["Cabin"] = test_data["Cabin"].isnull().apply(lambda x: 0 if x else 1) 
Test_data_clean["Sex"] = test_data["Sex"].apply(lambda x: 1 if x =='female' else 0)
Test_data_clean["First_class"] = test_data["Pclass"].apply(lambda x: 1 if x == 1 else 0)
Test_data_clean["Second_class"] = test_data["Pclass"].apply(lambda x: 1 if x == 2 else 0)
Test_data_clean["Age"] = test_data["Age"]
mean_age = Test_data_clean["Age"].mean()
Test_data_clean["Age"].fillna(mean_age, inplace=True)


# normalize the test data

Test_data_clean_normalized = scale.fit_transform(Test_data_clean)


# make prediction
prediction = model.predict(Test_data_clean_normalized)

# construct a Data Frame
mysubmission = pd.DataFrame({ 'Survived': prediction},index = test_data['PassengerId'])

# Save Data Frame as csv file
# mysubmission.to_csv('Submission.csv')


model = RandomForestClassifier(random_state=0)
model.fit(X_train,y_train)
print()
print("The score of the Random Forest model on validation set. {} ".format(model.score(X_test,y_test)))
print()
