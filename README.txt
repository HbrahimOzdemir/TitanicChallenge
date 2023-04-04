Titanic Cruise Ship Data Analysis Project
This project is designed to analyze data from the Titanic Cruise Ship dataset, a popular machine learning dataset. The project will try to determine the survival probabilities of the passengers on Titanic crash and examine the factors affecting the survival rate of the passengers.

Dataset
The project runs on two CSV files:

train.csv: The training dataset to use for training the model.
test.csv: The test dataset to use for testing the model.
Both datasets contain data on passengers aboard the Titanic Cruise Ship. These data include:

PassengerId: Passenger's identification number.
Survived: Survival status (0 = Not Survived, 1 = Survived).
Pclass: Passenger class (1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class).
Name: The name of the passenger.
Sex: The gender of the passenger.
Age: The age of the passenger.
SibSp: Number of siblings/spouses on board the passenger.
Parch: The passenger's number of parents/children on board.
Ticket: The ticket number of the passenger.
Mouse: The fare paid by the passenger.
Cabin: The passenger's cabin number.
Embarked: Where the passenger boarded the ship.

The lines titanicdata= pd.read_csv("train.csv") and titanictest = pd.read_csv("test.csv") are reading the training and test data. Data is read from a CSV file and converted to a pandas DataFrame object.

The barPlot() function generates bar graphs of the frequency distribution of the dataset's categorical features such as 'Sex', 'Pclass', 'Embarked', 'SibSp' and 'Parch'. The plotHist() function generates histograms of the distribution of numeric features such as 'Age' and 'Charge'. The script also generates a heatmap using the heatmap() function from seaborn.

The plotHist() function plots a histogram for a numeric feature. First, we calculate the frequency distribution of the feature and then visualize this distribution in a histogram.

The variables pclass_survival, gender_survival, sibsp_survival, and parch_survival calculate survival rates based on different characteristics of the training data.

The detect_outliers() function detects outliers in numeric features in the training data.

The script then continues to preprocess the dataset, leaving some features that are not required for analysis. The drop() function from pandas is used to drop the 'Ticket' and 'Cabin' properties from both the training and test sets. The str.extract() function from Pandas is used to extract the names of the passengers from the 'Name' property, which are then mapped to numeric values using a dictionary. Then the 'Name' and 'PassengerId' properties are removed from the training set and the 'Name' property from the test set.

Missing values in the 'age' attribute are then calculated using the median age of passengers of the same sex and class. The 'age' attribute is then aggregated into 5 bins using pandas' cut() function. Finally, the 'Gender' attribute maps to numeric values where 1 represents female and 0 represents male.

Finally, the distributions of different features of the training data are visualized using the barPlot() and plotHist() functions.Then submission.to_csv('submissions.csv', index=False) code is used to print the data inside a DataFrame named submission to a file named comolkko.csv in CSV file format. The index=False parameter is used to exclude the index column from the file.