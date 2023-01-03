from collections import Counter
from sklearn.metrics import f1_score, hamming_loss
import matplotlib.pyplot as plt

from collections import defaultdict

import numpy as np
import operator

from sklearn.preprocessing import MultiLabelBinarizer

import logging

def visualize(type_count, mlb, num_tag):
    plt.figure(figsize = (10,10))
    plt.barh(y = list(mlb.inverse_transform(np.reshape([1]*num_tag,(-1,num_tag)))[0]), width = type_count)
    plt.show()

def n_tag(y_pred,y_actual,right_type_count,wrong_type_count):
    _w = 0
    _r = 0
    for index in range(len(y_pred)):
        if (y_pred[index] != y_actual[index] and (y_pred[index] == 1 or y_actual[index] == 1)):
            _w +=1
            wrong_type_count[index] +=1
        elif (y_pred[index] == y_actual[index] and y_pred[index] == 1):
            _r +=1
            right_type_count[index] +=1
            
    return _w,_r,right_type_count, wrong_type_count
    
def label_transform(predicted_set, threshold, mlb, num_tag):
    cnt = 0
    result = [0] * len(predicted_set)
    for i in range(len(predicted_set)):
        t_set = [int(predicted_set[i][index] >= threshold) for index in range(len(predicted_set[i]))]
        result[i] = list(mlb.inverse_transform(np.reshape(t_set, (-1, num_tag)))[0])
        if (result[i] == []):
            top = np.argmax(predicted_set[i])
            t_set = [0] * num_tag
            t_set[top] = 1
            result[i] = list(mlb.inverse_transform(np.reshape(t_set, (-1, num_tag)))[0])
            cnt+=1

    logging.info("{} sample are hard to classify cause of low percent".format(cnt))

    return result

def apriori(data, sp_T = 0.5):
    init = []
    for i in data:
        for q in i:
            if(q not in init):
                init.append(q)
    init = sorted(init)
    s = int(len(init) * sp_T)
    c = Counter()
    for i in init:
        for d in data:
            if(i in d):
                c[i]+=1
    l = Counter()
    for i in c:
        if(c[i] >= s):
            l[frozenset([i])]+=c[i]
    pl = l
    pos = 1
    for count in range (2,1000):
        nc = set()
        temp = list(l)
        for i in range(0,len(temp)):
            for j in range(i+1,len(temp)):
                t = temp[i].union(temp[j])
                if(len(t) == count):
                    nc.add(temp[i].union(temp[j]))
        nc = list(nc)
        c = Counter()
        for i in nc:
            c[i] = 0
            for q in data:
                temp = set(q)
                if(i.issubset(temp)):
                    c[i]+=1
        l = Counter()
        for i in c:
            if(c[i] >= s):
                l[i]+=c[i]
        if(len(l) == 0):
            break
        pl = l
        pos = count
    return pl

def IoU(keys,values):
    tags = dict()
    for index in range(len(keys)):
        tags[str(index)] = {'tags' : keys[index], 'value' : values[index]}
    maxValue = max(tags.items(), key = lambda x: x[1]['value'])
    res = list()
    for key, value in tags.items():
        if (value['value'] == maxValue[1]['value']):
            res.append(value['tags'])

    temp = set(res[0])
    for index in range(1,len(res)):
        temp = temp.union(set(res[index]))
   
    return list(temp)

def predict_problem(args, Y_pred, Y_actual, dataset, mlb, verbose = True):
    #Initialization
    num_tag = args["apriori"]["num_tag"]

    apriori_set = label_transform(Y_pred, args["apriori"]["classify_T"], mlb, num_tag)
    
    
    # Custom Accuracy
    problem_items = defaultdict()
    problem_count = defaultdict()
    problem_check = defaultdict()
    problem_sum = defaultdict()

    for index in range(len(apriori_set)):
        pos = dataset['problem'][index]
        problem_items[pos] = []
        problem_check[pos] = np.zeros(Y_actual[0].shape).astype(int)
        problem_sum[pos] = np.zeros(Y_actual[0].shape).astype(float)
        problem_count[pos] = 0

    for index in range(len(apriori_set)):
        pos = dataset['problem'][index]
        if apriori_set[index] != []:
            problem_items[pos].append(apriori_set[index])
        problem_sum[pos] = np.sum([problem_sum[pos],Y_pred[index]], axis = 0)
        problem_check[pos] = Y_actual[index]
        problem_count[pos] +=1
    for value in problem_sum:
        problem_sum[value]= np.divide(problem_sum[value],problem_count[value])

    na_cnt = 0
    for prob_name in problem_items:
            result = dict(apriori(problem_items[prob_name]))
            if (result != {}):
                keys = [list(k) for k in list(result.keys())]
                values = [v for v in list(result.values())]
                t = list()
                t.append(IoU(keys,values))
                if (t == [[]]):
                    indexed = list(enumerate(problem_sum[prob_name]))
                    top = np.argmax(problem_sum[prob_name])
                    problem_items[prob_name] =  np.zeros(shape=(num_tag), dtype = np.int64)
                    problem_items[prob_name][top] = 1
                else:
                    t = mlb.transform(t)             
                    problem_items[prob_name] = t[0]
            else:
                na_cnt +=1
                indexed = list(enumerate(problem_sum[prob_name]))
                top = np.argmax(problem_sum[prob_name])
                problem_items[prob_name] =  np.zeros(shape=(num_tag), dtype = np.int64)
                problem_items[prob_name][top] = 1

    logging.info("{} samples that apriori algorithm not able to handle".format(na_cnt))


    predict_set = list()
    check_set = list()
    wrong = 0
    right = 0
    right_type_count = [0] * num_tag
    wrong_type_count = [0] * num_tag

    for prob_name in problem_items:
        predict_set.append(problem_items[prob_name])
        check_set.append(problem_check[prob_name])   
        if verbose:
            w, r, right_type_count, wrong_type_count= n_tag(problem_items[prob_name],problem_check[prob_name],right_type_count,wrong_type_count)
            wrong+=w
            right+=r

    if verbose:   
        visualize(right_type_count, mlb, num_tag)
        visualize(wrong_type_count, mlb, num_tag)
        logging.info("{} wrong sample, {} right sample".format(wrong, right))
    return predict_set, check_set