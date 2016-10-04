# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import subprocess
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.externals.six import StringIO

def get_game_data():
	 df = pd.read_csv("character-predictions_test.csv",  header=0, index_col=0)
	 return df

df = get_game_data()
df = df.drop_duplicates() #刪除重複值
n_samples = len(df) 
y = targets = df["isAlive"].values #目標

df_house = pd.get_dummies(df['house'])
df_title = pd.get_dummies(df['title'])
df_cul = pd.get_dummies(df['culture'])
df_new = pd.concat([df, df_house, df_title, df_cul], axis=1)


columns = list(df_new.columns[4:])

x = features = df_new[columns].values
clf = DecisionTreeClassifier(criterion="entropy", min_samples_split=30, max_depth=7)
clf = clf.fit(x[:n_samples*0.75], y[:n_samples*0.75])

expected = y[n_samples*0.75:]
predicted = clf.predict(x[n_samples*0.75:])
print ("expected: ",expected[:10])
print ("predicted:",predicted[:10])
# print(sum(predicted == expected))

print ("Confusion Matrix:\n",metrics.confusion_matrix(expected,predicted))
print ("Accuracy:\n",metrics.accuracy_score(expected,predicted))
print ("Confusion report for classifier \n \n",metrics.classification_report(expected,predicted))


with open("game.dot", 'w') as f:
  f = export_graphviz(clf, out_file=f, feature_names=columns)

command = ["dot", "-Tpng", "game.dot", "-o", "game.png"]
subprocess.check_call(command)


# def visualize_tree(tree, feature_names):
#     with open("dt.dot", 'w') as f:
#         export_graphviz(tree, out_file=f, feature_names=feature_names)
    
#     command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
#     try:
#         subprocess.check_call(command)
#     except:
#         exit("Could not run dot, ie graphviz, to produce visualization")

# visualize_tree(clf, columns)







# from sklearn.preprocessing import Imputer #如果有缺值的話 就把那一列的數值做個平均 http://christianherta.de/lehre/dataScience/machineLearning/decision-trees.php
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# X = imp.fit_transform(features)

