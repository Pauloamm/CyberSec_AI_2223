# This is a sample Python script.
#from sklearn import
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Classification Models
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

def DataReading():

    # Reading file
    datasetName = "Phishing_Websites.arff"
    datasetFolder = "Dataset"
    datasetPath = os.path.join(os.getcwd(),datasetFolder,datasetName)

    data = arff.loadarff(datasetPath)
    df = pd.DataFrame(data[0]).astype(int)

    return df


def ModelsTraining(datasetForTraining):

    # Splits info
    columnNames = datasetForTraining.columns

    xValues = datasetForTraining[columnNames[:-1]]
    yValues = datasetForTraining[columnNames[-1]]

    classificationModelsDictionary = \
        {
            1: (LinearSVC(), 'LinearSVC Model'),
            2: (KNeighborsClassifier(), 'KNeighborsClassifier Model'),
            3: (DecisionTreeClassifier(), 'DecisionTreeClassifier Model'),
            4: (RandomForestClassifier(), 'RandomForestClassifier Model'),
            5: (AdaBoostClassifier(), 'AdaBoostClassifier Model'),
            6: (MLPClassifier(), 'MLPClassifier Model'),

        }

    percentageForTraining = 0.7

    xValuesToTrain, xValuesToTest, yValuesToTrain, yValuesToTest = train_test_split(xValues, yValues, test_size=(
            1 - percentageForTraining), random_state=1)

    numberOfLines = 3
    numberOfColumns = 2

    #fig, axs = plt.subplots(numberOfLines, numberOfColumns)
    #fig.suptitle("Quality According to {} using:".format(columnNames[bestColumnToTestDependency]))

    for counter in range(1, len(classificationModelsDictionary) + 1):
        model = classificationModelsDictionary[counter][0]

        model.fit(xValuesToTrain, yValuesToTrain)

        yValuesPrediction = model.predict(xValuesToTest)

        print("\nClassification Report of {}".format(classificationModelsDictionary[counter][1]))
        #print(classification_report(yValuesToTest, yValuesPrediction, zero_division=1))
        print("Accuracy: " +str(accuracy_score(yValuesToTest,yValuesPrediction)) + "\n\n")

    #plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    datasetForTraining = DataReading()

    ModelsTraining(datasetForTraining)

