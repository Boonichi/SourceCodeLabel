import sys
import numpy as np
import random
import torch
import logging
import json
import time
from datetime import timedelta
import pickle
import os
from collections import defaultdict
from subprocess import *

def exit_handler(path,dataset):
    dump_dataset(path, dataset)

def syscmd_without_logging(cmd, encoding=''):
    """
    Runs a command on the system, waits for the command to finish, and then
    returns the text output of the command. If the command produces no text
    output, the command's return code will be returned instead.
    """
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
        close_fds=True)
    p.wait()
    output = p.stdout.read()
    if len(output) > 1:
        if encoding: return output.decode(encoding)
        else: return output
    return p.returncode

def ensure_path(path):
    path = str(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path
    
def run_system_command(cmd: str,
                       shell: bool = False,
                       err_msg: str = None,
                       verbose: bool = True,
                       split: bool = True,
                       stdout=None,
                       stderr=None) -> int:
    if verbose:
        sys.stdout.write("System cmd: {}\n".format(cmd))
    if split:
        cmd = cmd.split()
    rc = call(cmd, shell=shell, stdout=stdout, stderr=stderr)
    if err_msg and rc:
        sys.stderr.write(err_msg)
        exit(rc)
    return rc
    
def fcall(fun):
    """
    Convenience decorator used to measure the time spent while executing
    the decorated function.
    :param fun:
    :return:
    """
    def wrapper(*args, **kwargs):

        logging.info("[{}] ...".format(fun.__name__))

        start_time = time.perf_counter()
        res = fun(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time

        logging.info("[{}] Done! {}s\n".format(fun.__name__, timedelta(seconds=runtime)))
        return res

    return wrapper
    
def set_random_seed(args):
    if "torch" in sys.modules:
        torch.manual_seed(args["random_seed"])
    np.random.seed(int(args["random_seed"]))
    random.seed(args["random_seed"])

def setup_logging(args):
    
    level = {
        "info" : logging.INFO, 
        "debug" : logging.DEBUG,
        "critical" : logging.CRITICAL
    }
    
    msg_format = '%(asctime)s:%(levelname)s: %(message)s'
    formatter = logging.Formatter(msg_format, datefmt = '%H:%M:%S')
    args = args["logging"]

    file_handler = logging.FileHandler(args["filename"], mode = args["filemode"])
    file_handler.setLevel(level=level[args["level"]])
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger()
    logger.setLevel(level[args["level"]])

def load_dataset(path, verbose = True):
    if (verbose):
        logging.info("Load dataset {}!".format(path))
    path = str(path)
    
    if ".json" in path:
        with open(path, encoding = "utf-8") as f:
            if ".jsonl" in path:
                data = [json.loads(line) for line in f]
            elif ".json" in path:
                content = f.read()
                data = json.loads(content)
    elif ".pickle" in path:
        with open(path, "rb") as f :
            data = pickle.load(f)
    elif ".pt" in path:
        data = torch.load(path)
    else:
        raise NotImplementedError("Don't know how to load a dataset of this type")
    if (verbose):
        logging.info("Loaded {} records!".format(len(data)))
    return data

def dump_dataset(path,data, verbose = True):
    if verbose:
        print("Dump Dataset {}: {}!".format(path, len(data)))
    
    def dump_data():
        if '.json' in path:
            with open(path, 'w') as f:
                json.dump(data, f)
        elif '.pickle' in path:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
    
    path = str(path)
    try:
        dump_data()
    except FileNotFoundError:
        directory_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok= True)
            dump_data()

def dump_file(path, data):
    path = str(path)
    try:
        with open(path, "w") as f:
            f.write(data)
    except FileNotFoundError:
        directory_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok= True)
            with open(path, "w") as f:
                f.write(data)
@fcall
def parser_config(path):
    
    return load_dataset(path)

def save_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    f.close()

def ensure_path(path):
    path = str(path)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def checkSparse(dataset): 
    def isSparse(array) :
        
        counter = 0
    
        # Count number of zeros
        # in the matrix
        for i in range(len(array)) :
            if (array[i] == 0) :
                counter = counter + 1
    
        return (counter == len(array))
    cnt = 0
    for sample in dataset:
        if (isSparse(sample)):
            cnt+=1

    print(cnt)

def tf_gpu_housekeeping():
    """
    Ensure tensorflow doesn't hog the available GPU memory.
    :return:
    """
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logging.critical(str(e))
            exit(1)