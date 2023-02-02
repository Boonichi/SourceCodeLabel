from util import fcall, dump_dataset, load_dataset, load_problem_folder

from datasets.source_dataset import SourceDataset
from datasets.path_dataset import PathDataset
from datasets.dataset import Dataset

import os
from collections import defaultdict
import logging

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np

import atexit
from util import exit_handler

from tqdm import tqdm

import random

import pandas as pd

def identify_handler(handlers):
    handler_map = {
        "source" : SourceDataset,
        "path" : PathDataset
    }
    try:
        return [handler_map[handler] for handler in handlers.split("|")]
    except KeyError:
        raise NotImplementedError("Not Recognized the handler type")

def TagLoader(args):
    tag_removed_list = ['2-sat', 'chinese remainder theorem','fft','schedules','meet-in-the-middle','ternary search','expression parsing','graph matchings',"string suffix structures","binary search"]
    search_list = ["meet-in-the-middle","binary search","ternary search"]
    graph_list = ["graph matchings"]
    math_list = ["fft","chinese remainder theorem"]
    string_list = ["expression_parsing", "string suffix structures"]

    tags = list()
    Tagset = dict()
    with open(args["tags"],'r') as file:
        content = file.readlines()
        for line in content:
            line = line.rstrip()
            problem = list(eval(line).items())[0][0]
            tags = list(eval(line).items())[0][1]
            tag_result = set()
            if problem in os.listdir(args["dataset"]):
                for tag in tags:
                    if tag in tag_removed_list: 
                        if tag in search_list:
                            tag_result.add("search technique")
                        elif tag in graph_list:
                            tag_result.add("graphs")
                        elif (tag in math_list):
                            tag_result.add("math")
                        elif (tag in string_list):
                            tag_result.add("strings")
                    else: tag_result.add(tag)
                tag_result = list(tag_result)
                Tagset[problem] = tag_result
    return Tagset
    
def tag_distribution():
    return 0

def tag_filter(dataset, Tagset):
    result = list()
    for sample in Dataset:
        if sample["problem"] in list(Tagset.keys()) and Tagset[sample["problem"]] != []:
            result.append(sample)
    return sample

@fcall
def create_raw_dataset(args):
    raw_path = args["dataset"]
    dataset = list()
    if (args["TagState"]):
        Tagset = load_dataset(args["tag_prepared"])
    else:
        Tagset = TagLoader(args)
        dump_dataset(args["tag_prepared"], Tagset)
    
    dataset = load_problem_folder(raw_path)
    dataset = tag_filter(dataset, Tagset)

    new_raw_path = args["raw_dataset"]
    dump_dataset(new_raw_path, dataset)         

def prepare_dataset_parallel(args, dataset, handlers):
    if "source" in args["prepare"]["handler"]:
        dataset = SourceDataset(args, dataset).prepare()
    if "path" in args["prepare"]["handler"]:
        dataset = PathDataset(args, dataset).prepare()
    return dataset
@fcall 
def preprocessing(args):
    new_path = args["prepared_dataset"]

    if (args["prepare"]["ContinueState"]):
        dataset_path = args["prepared_dataset"]
    else:
        dataset_path = args["raw_dataset"]
    dataset = load_dataset(dataset_path)

    atexit.register(exit_handler, new_path, dataset)

    handlers = identify_handler(args["prepare"]["handler"])
    if (args["prepare"]["ContinueState"] == False):
        dataset = Dataset(args, dataset).prepare()
    if (args['prepare']["parallel"]):
        dataset = prepare_dataset_parallel(args, dataset, handlers)
    else:
        for handler in handlers:
            logging.info("Start {}".format(str(handler)))
            dataset = handler(args,dataset).prepare()

    dump_dataset(new_path, dataset)

def set_limit(args, content, type):
    MAX_CONTEXT = args["split"]["MAX_CONTEXT"][type]
    if (len(content) > MAX_CONTEXT):
        content = random.sample(content, MAX_CONTEXT)
    return content

def extract_path_based(args, path, type):
    sources = []
    paths = []
    values = []
    with open(path, "r") as f:
        content = f.readlines()
        content = set_limit(args, content, type)

    for line in content:
        source, path, value = line.split(",")
        sources.append(source)
        paths.append(path)
        values.append(value)
    
    return sources, paths, values

def load_cpg_path(args, dataset, path : str):
    result = []
    types = ["ast", "cfg", "cdg", "pdg"]
    for index, sample in tqdm(enumerate(dataset)):
        source_path = os.path.join(path,sample["problem"], sample["index"])
        count = 0   
        for file in os.listdir(source_path):
            name = file.split(".")[0]
            if (file.endswith(".txt")):
                sources, paths, values = extract_path_based(args, os.path.join(source_path, file), name)
                sample[name + "_source"] = sources
                sample[name + "_path"] = paths
                sample[name + "_value"] = values
                count +=1
        if (count == len(types)):
            result.append(sample)
    logging.info("Loaded {} samples in CPG Dataset".format(len(result)))
    return result

@fcall
def split_dataset(args):
    cpg_path = args["cpg_dataset"]
    
    if (args["split"]["ContinueState"] == False):
        dataset = load_dataset(args["prepared_dataset"])
        dataset = load_cpg_path(args, dataset, cpg_path)
        dump_dataset(args["cpg"],dataset, verbose = False)
    else:
        dataset = load_dataset(args["cpg"])
    train, test = split_stratified(args,dataset)
    
    data_split = {
        "train" : train,
        "test" : test,
    }
    
    for key in data_split:
        dump_dataset(args[key+"_dataset"] + key + ".pickle", data_split[key])

def split_stratified(args,dataset):
    problems = [[sample["problem"], sample["tags"]] for sample in dataset]
    X = set()
    Y = list()
    for sample in problems:
        if sample[0] not in X:
            X.add(sample[0])
            Y.append(sample[1])
    
    X = np.array(list(X))
    Y = np.array(Y)
    
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y)
    
    Percent = args["split"]["train"]
    msss = MultilabelStratifiedShuffleSplit(n_splits = 1, test_size = (1 - Percent))

    for train_index, test_index in msss.split(X,Y):
        train_problems, test_problems = X[train_index], X[test_index]
    
    train_set = []
    for item in dataset:
        if item['problem'] in train_problems:
            train_set.append(item)
    test_set = []
    for item in dataset:
        if item['problem'] in test_problems:
            test_set.append(item)
    train_set = pd.DataFrame(data = train_set)
    test_set = pd.DataFrame(data = test_set)
    print(train_set.shape, test_set.shape)
    
    return train_set, test_set

        