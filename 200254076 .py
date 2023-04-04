import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC, LinearSVC


titanicdata = pd.read_csv("train.csv") 
titanictest = pd.read_csv("test.csv") 
titanic = [titanicdata, titanictest] 

def barPlot(feature):
    values = titanicdata[feature].value_counts()
    plt.bar(values.index, values)
    plt.xticks(values.index, values.index.values)
    plt.ylabel("Frequency")
    plt.title(feature)
    plt.show()

# Create a histogram for a given numerical feature
def plotHist(feature):
    plt.hist(titanicdata[feature], bins=50)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(feature)
    plt.show()

titanicdata.describe(include=['O']) 
titanicdata[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False) 
titanicdata[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False) 
titanicdata[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False) 
titanicdata[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False) 

sns.heatmap(titanicdata.corr(),cmap="YlOrRd")
plt.show()



# Detect outliers in numerical features of train data
def detect_outliers(dataFrame, features):
    outlier_indices = []

    for c in features:
        Q1 = np.percentile(dataFrame[c], 25)
        Q3 = np.percentile(dataFrame[c], 75)
        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        outlier_list_col = dataFrame[(dataFrame[c] < Q1 - outlier_step) | (dataFrame[c] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = pd.Series(outlier_indices)
    multiple_outliers = outlier_indices.value_counts()[outlier_indices.value_counts() > 2].index.tolist()

    
    return multiple_outliers 

# Call the barPlot() function for each categorical feature
barPlot("Survived")
barPlot("Sex")
barPlot("Pclass")
barPlot("Embarked")
barPlot("SibSp")
barPlot("Parch")

# Call the plotHist() function for each numerical feature
plotHist("Age")
plotHist("Fare")


titanicdata = titanicdata.drop(['Ticket', 'Cabin'], axis=1)
titanictest = titanictest.drop(['Ticket', 'Cabin'], axis=1)


titanic = [titanicdata, titanictest]


for dataset in titanic:
    dataset['Header'] = dataset['Name'].str.extract(" ([A-Za-z]+)\,", expand=False)


pd.crosstab(titanicdata['Header'], titanicdata['Sex'])



mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}


for dataset in titanic:
    dataset['Header'] = dataset['Header'].map(mapping)
    dataset['Header'] = dataset['Header'].fillna(0)

titanicdata = titanicdata.drop(['Name', 'PassengerId'], axis=1)
titanictest = titanictest.drop(['Name'], axis=1)

titanic = [titanicdata, titanictest]

for dataset in titanic:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

aver_ages = np.zeros((2, 3))

# Iterate through each dataset 
for dataset in [titanicdata, titanictest]:
    
  
    for sex in [0, 1]:  # 0 is for male and 1 is for female
        for pclass in [1, 2, 3]:
            
        
            median_age = dataset[(dataset['Sex'] == sex) & (dataset['Pclass'] == pclass)]['Age'].dropna().median()
            
          
            if np.isnan(median_age):
                continue
            
        
            rounded_age = round(median_age * 2) / 2
            
       
            dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] == sex) & (dataset['Pclass'] == pclass), 'Age'] = rounded_age
            
  
    dataset['Age'] = dataset['Age'].astype(int)


titanicdata.head(10)
titanicdata['AgeBand'] = pd.cut(titanicdata['Age'], 5)
titanicdata[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


for dataset in titanic:
    dataset['AgeBand'] = dataset['Age']

titanicdata = titanicdata.drop(['Age'], axis=1)
titanictest = titanictest.drop(['Age'], axis=1)
titanic = [titanicdata, titanictest]

for dataset in titanic:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

titanicdata[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in titanic:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


titanicdata = titanicdata.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
titanictest = titanictest.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
titanic = [titanicdata, titanictest]


freq_port = titanicdata.Embarked.dropna().mode()[0]
for dataset in titanic:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

titanicdata[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',ascending=False)

for dataset in titanic:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

titanictest['Fare'].fillna(titanictest['Fare'].dropna().median(), inplace=True)
titanictest.head()
titanicdata['FareBand'] = pd.qcut(titanicdata['Fare'], 4)
titanicdata[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',ascending=True)

for dataset in [titanicdata, titanictest]:
    
    
    fare_bins = [0, 7.91, 14.454, 31, np.inf]
    fare_labels = [0, 1, 2, 3]  
    
    
    dataset['Fare'] = pd.cut(dataset['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)
    
    
    dataset['Fare'] = dataset['Fare'].astype(int)



titanicdata = titanicdata.drop(columns=['FareBand'])


titanic = [titanicdata, titanictest]


X_data = titanicdata.drop(columns=["Survived"])
Y_data = titanicdata["Survived"]


X_test = titanictest.drop(columns=["PassengerId"]).copy()

svc = SVC()
svc.fit(X_data, Y_data)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_data, Y_data) * 100, 2)


submission = pd.DataFrame({
    "PassengerId": titanictest["PassengerId"],
    "Survived": Y_pred
})


submission.to_csv('submission.csv', index=False)
