from sklearn.metrics import f1_score, hamming_loss
import logging

def F1_score_micro(problem_pred, problem_actual):
    logging.info("F1 Score Micro : {}".format(f1_score(problem_pred, problem_actual, average = "micro")))

def F1_score_macro(problem_pred, problem_actual):
    logging.info("F1 Score Micro : {}".format(f1_score(problem_pred, problem_actual, average = "macro")))
  

def hamming(problem_pred, problem_actual):    
    logging.info("Hamming Loss : {}".format(hamming_loss(problem_pred, problem_actual)))
