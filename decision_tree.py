#-------------------------------------------------------------------------
# AUTHOR: Sidharth Basam
# FILENAME: decision_tree.py
# SPECIFICATION: Reads contact_lens.csv and build a decision tree
# FOR: CS 4210- Assignment #1
# TIME SPENT: 5 hours
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.

# Importing Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv

db = []
X = []
Y = []




#read the data from csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skipping the header
            db.append(row)

#define categorical mappings with the different attributes on table
ageMap = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
spectacleMap = {"Myope": 1, "Hypermetrope": 2}
astigmatismMap = {"No": 1, "Yes": 2}
tearMap = {"Reduced": 1, "Normal": 2}
lensMap = {"Yes": 1, "No": 2}

#turn categorical training features into numbers and add to the 4D array 
X = [[ageMap[row[0]], spectacleMap[row[1]], astigmatismMap[row[2]], tearMap[row[3]]] for row in db]

#turn the training classes into numbers and add to vector Y
Y = [lensMap[row[4]] for row in db]

# fit the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

# create the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes', 'No'], filled=True, rounded=True)
plt.show()
