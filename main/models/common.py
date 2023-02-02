from models.embeddings.embedding import Tokenizer, BertTokenizer

from util import load_dataset, dump_dataset, fcall, tf_gpu_housekeeping

from models.classifiers.sourcegraph import SourceGraph

from models.apriori import predict_problem

from models.metrics import *

import tensorflow as tf
import logging

def identify_tokenizer(args):
    tokenizers = {
        "Bert" : BertTokenizer,
        "Default" : Tokenizer 
    }
    return tokenizers[args["embedding"]["tokenizer"]]


@fcall
def prepare_input(args):
    train_set = load_dataset(args["train_dataset"] + "train.pickle")
    test_set = load_dataset(args["test_dataset"] + "test.pickle")

    tokenizer = identify_tokenizer(args)
    tokenizer = tokenizer(args)
    X_train, Y_train = tokenizer.run(train_set, "train")
    X_test, Y_test = tokenizer.run(test_set, "test")
    
    dump_split = {
        "X_train" : X_train,
        "Y_train" : Y_train,
        "X_test" : X_test,
        "Y_test" : Y_test
    }

    for key in dump_split:
        dump_dataset(args[key.split("_")[1]+ "_dataset"] + key + ".pickle", dump_split[key])

def setup_model(args, ContinueState = False):
    model = SourceGraph(args)
    model.build_model(model_name = "SourceGraph")
    if ContinueState:
        model.load_weights()

    return model

def load_input(args, fold):
    logging.info("Loading preprocessed {} dataset".format(fold))
    X = load_dataset(args[fold + "_dataset"] + "X_{}.pickle".format(fold))
    Y = load_dataset(args[fold + "_dataset"] + "Y_{}.pickle".format(fold))
    
    return X, Y


@fcall
def train(args):

    tf_gpu_housekeeping()

    model = setup_model(args, ContinueState = args["train"]['ContinueState'])   
    train_set = load_input(args, "train")
    test_set = load_input(args, "test")
    
    model.fit(train_set, test_set)

@fcall
def test(args):

    model = setup_model(args, ContinueState = True)
    test_raw = load_dataset(args["test_dataset"] + "test.pickle")
    test_set = load_input(args, "test")
    mlb = load_dataset(args["mlb"], verbose = False)

    source_pred = model.predict(test_set)
    problem_pred, problem_actual = predict_problem(args, source_pred, test_set[1], test_raw, mlb)

    #Print Random Sample
    #if args["evaluation"]["sample"]:
    #    logging.info("{}".format(0))
    
    #Metrics
    metrics = {
        "F1_score_micro" : F1_score_micro,
        "F1_score_macro" : F1_score_macro,
        "hamming_loss" : hamming
    }

    for fun in args["evaluation"]["metrics"].split("|"):
        metrics[fun](problem_pred, problem_actual)