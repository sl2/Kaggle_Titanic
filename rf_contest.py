#coding: utf-8
import numpy as np
import csv as csv
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import pylab as pl


def read_titled_csv(filepath):
    csv_file = csv.reader(open(filepath,'rb'))
    header = csv_file.next()
    data = []
    for row in csv_file:
        data.append(row)
    return header, np.array(data)

def get_titanic_train(data):
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

    for row in data:
        if row[11] != '' and row [5] != '':
            row[4] = sex[row[4]]
            row[11] = embarked[row[11]]
            #Pclass,Sex,Age,SibSp,Parch,Embarked
            a = np.hstack((np.hstack((row[2],row[4:8])),row[11]))
            a = a.astype(np.float)
            feature.append(a)
            label.append(row[1])
    return feature, label

# Split Train Data to Train Data and Test Data
def get_train_test(feature, label, train_count, test_count):
    train = {
        'feature':feature[0:train_count],
        'label':label[0:train_count]
    }
    test = {
        'feature':feature[train_count:train_count+test_count],
        'label':label[train_count:train_count+test_count]
    }
    return train, test

def get_score(train):
    score = clf.score(train['feature'], train['label'])
    cross_score = cross_val_score(clf, np.array(train['feature']), np.array(train['label']))
    return {"score":score, "cross_score":cross_score.mean()}

def get_predict(test):
    predict = []
    for row in test['feature']:
        predict.append(clf.predict(row)[0])
    return predict

def get_result(test):
    predict = get_predict(test)

    correct = 0
    incorrect = 0
    for i in range(len(test['feature'])):
        if predict[i] == test['label'][i]:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
    
    ratio = (correct / float(correct + incorrect)) * 100
    return {"correct:":correct, "incorrect":incorrect, "ratio":ratio}


def visualize(data):
    feature = data['feature']
    label = data['label']
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

    train_file = './train/train.csv'
    header, data = read_titled_csv(train_file)

    feature, label = get_titanic_train(data)

    train_count = 600
    test_count = 100

    train, test = get_train_test(feature, label, train_count, test_count)
    clf = RandomForestClassifier(
            n_estimators=1000, 
            max_features=None, 
            bootstrap=True
            #n_jobs=1
        )
    clf.fit(train['feature'], train['label'])
    print get_score(train)
    print get_result(test)

    visualize(train)
    visualize(test)

