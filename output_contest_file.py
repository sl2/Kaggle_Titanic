#coding: utf-8

# Kaggle Titanic
# Sample code for "Titanic: Machine Learning from Disaster" contest of Kaggle
#   - elementary prediction by random forest
#
# MIT Licensec, 2013 so
#

import numpy as np
import csv as csv
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import pylab as pl
from collections import Counter
from datetime import datetime


"""
Read header and rows from csv that is provided by Kaggle.
"""
def read_titled_csv(filepath):
    csv_file = csv.reader(open(filepath,'rb'))
    header = csv_file.next()
    data = []
    for row in csv_file:
        data.append(row)
    return header, np.array(data)

"""
Get feature vectors and labels from result of read_titled_csv(). 

Option:
    (Bool) remove_empty: Remove rows that has empty fields.
"""
def get_titanic_train(data, remove_empty):
    """ 
    0:'PassengerId'
    1:'Survived'
    2:'Pclass'
    3:'Name'
    4:'Sex'
    5:'Age'
    6:'SibSp'
    7:'Parch'
    8:'Ticket'
    9:'Fare'
    10:'Cabin'
    11:'Embarked'
    """
    sex = {'female':0,'male':1}
    embarked = {'C':0,'Q':1,'S':2}
    
    feature = []
    label = []

    if remove_empty: # remove
        for row in data:
            if row[5] != '' and row[11] != '':
                row[4] = sex[row[4]]
                row[11] = embarked[row[11]]
                #Pclass,Sex,Age,SibSp,Parch,Embarked
                a = np.hstack((np.hstack((row[2],row[4:8])),row[11]))
                a = a.astype(np.float)
                feature.append(a)
                label.append(row[1])
    else:
        age_list = [float(row[5]) for row in data if row[5] != '']
        embarked_list = [row[11] for row in data if row[11] != '']
        average_age = sum(age_list)/len(age_list)
        most_common_embarked, count_most_common_embarked = Counter(embarked_list).most_common()[0]

        for row in data:
            if row [5] == '':
                row[5] = average_age
            if row[11] == '':
                row[11] = most_common_embarked
            row[4] = sex[row[4]]
            row[11] = embarked[row[11]]
            #Pclass,Sex,Age,SibSp,Parch,Embarked
            a = np.hstack((np.hstack((row[2],row[4:8])),row[11]))
            a = a.astype(np.float)
            feature.append(a)
            label.append(row[1])
    return feature, label


def get_titanic_contest(data):
    """ 
    0:'PassengerId'
    1:'Pclass'
    2:'Name'
    3:'Sex'
    4:'Age'
    5:'SibSp'
    6:'Parch'
    7:'Ticket'
    8:'Fare'
    9:'Cabin'
    10:'Embarked'
    """
    sex = {'female':0,'male':1}
    embarked = {'C':0,'Q':1,'S':2}
    
    feature = []
    label = []

    age_list = [float(row[4]) for row in data if row[4] != '']
    embarked_list = [row[10] for row in data if row[10] != '']
    average_age = sum(age_list)/len(age_list)
    most_common_embarked, count_most_common_embarked = Counter(embarked_list).most_common()[0]
    
    for row in data:
        if row[4] == '':
            row[4] = average_age
        if row[10] == '':
            row[10] = most_common_embarked
        row[3] = sex[row[3]]
        row[10] = embarked[row[10]]
        #Pclass,Sex,Age,SibSp,Parch,Embarked
        a = np.hstack((np.hstack((row[1],row[3:7])),row[10]))
        a = a.astype(np.float)
        feature.append(a)
    return feature

# Split Train Data to Train Data and Test Data
def create_test_data(feature, label, train_count, test_count):
    train = {
        'feature':feature[0:train_count],
        'label':label[0:train_count]
    }
    test = {
        'feature':feature[train_count:train_count+test_count],
        'label':label[train_count:train_count+test_count]
    }
    return train, test

def get_score(feature,label):
    score = clf.score(feature, label)
    cross_score = cross_val_score(clf, np.array(feature), np.array(label))
    return {"score":score, "cross_score":cross_score.mean()}

def get_predict(feature, classifier):
    predict = []
    for row in feature:
        predict.append(classifier.predict(row)[0])
    return predict

def get_result(predict, label):
    assert len(predict) == len(label)
    correct = 0
    incorrect = 0
    for i in range(len(predict)):
        if predict[i] == label[i]:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
    
    ratio = (correct / float(correct + incorrect)) * 100
    return {"correct:":correct, "incorrect":incorrect, "ratio":ratio}

def count_labels(predicted_data):
    return Counter(predicted_data).most_common()

def visualize(feature, label):
    survival_no = []
    survival_no_age = []
    survival_no_sex = []
    survival_no_embarked = []
    survival_yes = []
    survival_yes_age = []
    survival_yes_sex = []
    survival_yes_embarked = []

    for i in range(len(feature)):
        if label[i][0] == '0':
            survival_no.append(int(label[i][0]))
            survival_no_sex.append(int(feature[i][1]))
            survival_no_age.append(float(feature[i][2]))
            survival_no_embarked.append(int(feature[i][5]))
        elif label[i][0] == '1':
            survival_yes.append(int(label[i][0]))
            survival_yes_sex.append(int(feature[i][1]))
            survival_yes_age.append(float(feature[i][2]))
            survival_yes_embarked.append(int(feature[i][5]))

    pl.hist(survival_no_age, bins=20, histtype='step', normed=True)
    pl.hist(survival_yes_age, bins=20, histtype='step', normed=True)
    pl.show()

    pl.hist(survival_no_sex, bins=2, histtype='step', normed=True)
    pl.hist(survival_yes_sex, bins=2, histtype='step', normed=True)
    pl.show()

    pl.hist(survival_no_embarked, bins=3, histtype='step', normed=True)
    pl.hist(survival_yes_embarked, bins=3, histtype='step', normed=True)
    pl.show()



if __name__ == '__main__':
    print "Test..."

    # Read csv file.
    train_file = './train/train.csv'
    header, data = read_titled_csv(train_file)

    # Get train data.
    feature, label = get_titanic_train(data,True)

    # Create test data from train data.
    train_count = 600 # Number of train data
    test_count = 100 # Number of test data
    train, test = create_test_data(feature, label, train_count, test_count)
    
    # Define Ramdom Forest Classifier
    clf = RandomForestClassifier(
            n_estimators=100, 
            max_features=None, 
            #bootstrap=True
            bootstrap=True
            #n_jobs=1
        )
    assert len(train['feature']) == len(train['label'])
    print "Train Data:", len(train['feature'])
    clf.fit(train['feature'], train['label'])
    print get_score(train['feature'], train['label'])
    print "Feature importances:", clf.feature_importances_

    # Predict the label(answer) of test data
    print "Test Data:", len(test['feature'])
    predict = get_predict(test['feature'], clf)
    print "Summary of labels:", count_labels(predict)
    print get_result(predict, test['label'])
    
    # Visualize
    # visualize(train['feature'], train['label'])
    # visualize(test['feature'], test['label'])

    # Predict the label(answer) of contest data N times.
    print "\nCreat contest file..."
    result_list = []
    PREDICT_TIMES = 9
    for i in range(PREDICT_TIMES):
        # Define classifier
        clf1 = RandomForestClassifier(
                n_estimators=100,
                max_features=None, 
                bootstrap=True
            )
        assert len(feature) == len(label)
        clf1.fit(feature, label)
        print i, get_score(feature, label)
        print "Feature importances:", clf1.feature_importances_

        # Read contest data
        contest_file = './test/test.csv'
        contest_header, contest_data = read_titled_csv(contest_file)
        contest_feature = get_titanic_contest(contest_data)
        
        # Predict contest data
        contest_predict = get_predict(contest_feature,clf1)
        assert len(contest_predict) == len(contest_data)

        # Count number of each label 
        print "Summary of labels", count_labels(contest_predict)
        result_list.append(contest_predict)

    result_list_summary = []
    for i in range(len(contest_predict)):  
        label_sum = count_labels([x[i] for x in result_list])
        #if len(label_sum)>1:
        #    print "Need majority vote:", label_sum
        result_list_summary.append(label_sum)

    # Output csv file for contest
    d = datetime.now()
    dt = d.strftime('%Y%m%d_%H%M%S')
    output_file = './result/result_'+ dt +'.txt'
    f = open(output_file, 'w')
    f.write("passengerId, Survived\n") # Contest require a header row.
    for i in range(len(contest_data)):
        passengerid = str(contest_data[i][0])
        survived = str(result_list_summary[i][0][0]) #Decide by majority vote
        line = passengerid + "," + survived + "\n"
        f.write(line)
    print "Output:" + output_file 





