#coding: utf-8

# Kaggle Titanic
# Sample code for "Titanic: Machine Learning from Disaster" contest of Kaggle
#   - elementary prediction by random forest
#
# MIT Licensec, 2013 so


import numpy as np
import csv as csv
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
import pylab as pl
from collections import Counter
from datetime import datetime


SEX = {'female':0,'male':1}
EMBARKED = {'C':0,'Q':1,'S':2, '':-1}

"""
Read header and rows from csv that is provided by Kaggle.
"""
def read_titled_csv(filepath):
    csv_file = csv.reader(open(filepath,'rb'))
    header = csv_file.next()
    data = []
    for x in csv_file:
        data.append(x)
    return header, data

"""
Get feature vectors and labels from result of read_titled_csv(). 
"""
def format_titanic_train(data):
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
    feature = []
    label = []
    for x in data:
        a = [x[2]] + x[4:8] + [x[9]] + [x[11]]
        assert len(a) == 7
        feature.append(a)
        label.append(x[1])
    return feature, label

def format_titanic_contest(data):
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
    feature = []
    label = []
    for x in data:
        a = [x[1]] + x[3:7] + [x[8]] + [x[10]]
        assert len(a) == 7
        feature.append(a)
    return feature

# feature : [[FV_1 = Pclass, Sex, Age, SibSp, Parch, fare, Embarked],...,[FV_N]]
# label : [label_1,...,label_N]

# Fill empty age by average age
def fill_empty_age(feature):
    age_list = [float(x[2]) for x in feature if x[2] != '']
    average_age = float(sum(age_list)) / float(len(age_list))
    _feature = [] 
    for x in feature:
        if x[2] == '':
            x[2] = average_age
        _feature.append(x)
    return _feature

def fill_empty_fare(feature):
    fare_list = [float(x[6]) for x in feature if x[5] != '']
    average_fare = float(sum(fare_list)) / float(len(fare_list))
    _feature = [] 
    for x in feature:
        if x[5] == '':
            x[5] = average_fare
        _feature.append(x)
    return _feature

# Fill empty embarked by most common embarked
def fill_empty_embarked(feature):
    embarked_list = [x[6] for x in feature if x[6] != -1]
    most_common_embarked, count_most_common_embarked = Counter(embarked_list).most_common()[0]
    _feature = [] 
    for x in feature:
        if x[6] == -1:
            x[6] = most_common_embarked
        #Pclass,Sex,Age,SibSp,Parch,Embarked
        _feature.append(x)
    return _feature

def convert_to_num(feature):
    _feature = [] 
    for x in feature:
        x[1] = SEX[x[1]]
        x[6] = EMBARKED[x[6]]
        _feature.append(x)
    return _feature

# Predict empty age by other field
def predict_empty_age(feature):
    # Define Ramdom Forest Regressor
    rgr = RandomForestRegressor(
            n_estimators=1000,
            max_features=None, 
            bootstrap=True
        )
    train = []
    train_index = []
    label = []
    label_index = []
    test = []
    test_index = []
    for i, x in enumerate(feature):
        if x[2] == '':
            test.append(x[0:2] + x[3:])
            test_index.append(i)
        else:
            train.append(x[0:2] + x[3:])
            train_index.append(i)
            label.append(x[2])
            label_index.append(i)

    rgr.fit(train, label)

    predict = []
    for i, x in enumerate(test):
        age = rgr.predict(x)
        a = x[0:2] + [age[0]] + x[2:]
        predict.append(a)

    test_dic = dict(zip(test_index,predict))

    #Pclass, Sex, Age, SibSp, Parch, Embarked
    _feature = []
    for i, x in enumerate(feature):
        if i in test_index:
            a = test_dic[i]
            _feature.append(a)
        else:
            _feature.append(x)

    assert len(feature) == len(_feature)
    return _feature

# Predict empty embarked by other field
def predict_empty_embarked(feature):
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
    for x in feature:
        predict.append(classifier.predict(x)[0])
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
    return {"correct:":correct, "incorrect":incorrect, "correct ratio":ratio}

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
    feature, label = format_titanic_train(data)
    feature = convert_to_num(feature)
    feature = fill_empty_embarked(feature)
    feature = fill_empty_fare(feature)
    feature = predict_empty_age(feature)
    
    _feature = []
    for x in feature:
        _feature.append([float(y) for y in x])
    feature = _feature

    print "Number of feature vectors:", len(data)
    print "Number of valid feature vectors:", len(feature)

    # Create test data from train data.
    train_count = 600 # Number of train data
    test_count = 100 # Number of test data
    train, test = create_test_data(feature, label, train_count, test_count)
 
    # Define Ramdom Forest Classifier
    clf = RandomForestClassifier(
            n_estimators=1000,
            max_features=None, 
            bootstrap=True
            #n_jobs=1
        )
    assert len(train['feature']) == len(train['label'])
    print "Number of train data:", len(train['feature'])
    print "Number of test data:", len(test['feature'])
    clf.fit(train['feature'], train['label'])
    print get_score(train['feature'], train['label'])
    print "Feature importances:", clf.feature_importances_

    # Predict the label(answer) of test data
    print "Number of test data:", len(test['feature'])
    predict = get_predict(test['feature'], clf)
    print "Summary of labels:", count_labels(predict)
    print get_result(predict, test['label'])
    
    # Visualize
    #visualize(train['feature'], train['label'])
    #visualize(test['feature'], test['label'])

    # Predict the label(answer) of contest data N times.
    print "\nCreat contest file..."
    result_list = []
    PREDICT_TIMES = 5
    for i in range(PREDICT_TIMES):
        # Define classifier
        clf1 = RandomForestClassifier(
                n_estimators=1000,
                max_features=None, 
                bootstrap=True
            )
        assert len(feature) == len(label)

        print "Number of train data:", len(feature)
        clf1.fit(feature, label)
        print i, get_score(feature, label)
        print "Feature importances:", clf1.feature_importances_

        # Read contest data
        contest_file = './test/test.csv'
        contest_header, contest_data = read_titled_csv(contest_file)
        contest_feature = format_titanic_contest(contest_data)
        contest_feature = convert_to_num(contest_feature)
        contest_feature = fill_empty_embarked(contest_feature)
        contest_feature = fill_empty_fare(contest_feature)
        contest_feature = predict_empty_age(contest_feature)    
        
        _contest_feature = []
        for x in contest_feature:
            _contest_feature.append([float(y) for y in x])
        contest_feature = _contest_feature




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
    output_file = './result/result_'+ dt +'.csv'
    f = open(output_file, 'w')
    f.write("passengerId,Survived\n") # Contest require a header row.
    for i in range(len(contest_data)):
        passengerid = str(contest_data[i][0])
        survived = str(result_list_summary[i][0][0]) #Decide by majority vote
        line = passengerid + "," + survived + "\n"
        f.write(line)
    print "Output:" + output_file 


