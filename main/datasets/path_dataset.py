from util import dump_file, load_dataset, ensure_path
import os
import logging
from tqdm import tqdm
import shutil

from datasets.dataset import Dataset

from PathBasedProcess.cpg_extractor import CPG_Extractor

from threading import Thread
from multiprocessing import Process

import time

class PathDataset(Dataset):
    
    def __init__(self, args, data):
        super().__init__(args, data)
        self.workspace = "./storages/workspace/"
        self.savedDir = "./storages/cpgDataset/"        
        self.parallel = self.args["prepare"]["parallel"]
        self.num_worker = self.args["prepare"]["NumWorker"]
        self.num_source = self.args["prepare"]["NumSource"]

    def generate_path_based_multithreading(self):
        datafolds = chunkByProblem(self.data, self.num_worker)
        thread = [0] * self.num_worker
        logging.info("Starting multithreading generate path based dataset")
        for worker in range(self.num_worker):
            thread[worker] = Process(target = self.generate_path_based_singlethreading, args = (datafolds[worker], worker,))
            #thread[worker] = Thread(target = self.generate_path_based_singlethreading, args = (datafolds[worker], worker,))
            thread[worker].start()
        for worker in range(self.num_worker):
            thread[worker].join()

        #return dataset

    def generate_path_based_singlethreading(self, data, worker = None):
        print(len(data))
        for index, sample in tqdm(enumerate(data)):
            check = False

            input_path = os.path.join(self.workspace, "temp_{}.cpp".format(worker))
            output_path = os.path.join(self.savedDir, sample["problem"])
            
            for type in ['ast', 'cfg', 'cdg', 'ddg']:
                if type not in sample.keys(): 
                    if (os.path.exists(os.path.join(output_path, sample["index"], "{}.txt".format(type))) == False):
                        check = True
                        break

            if (check):
                source = sample["source"]

                dump_file(input_path, source)
                ensure_path(output_path)
                if len(os.listdir(output_path)) <= self.args["prepare"]["NumSource"]:
                    try:
                        ensure_path(os.path.join(output_path, sample["index"]))
                        CPG_Extractor(input_path = input_path,output_path=output_path, filename = sample["index"], WorkerIndex= worker).extract()
                    except:
                        shutil.rmtree(os.path.join(output_path,sample["index"]))
                        logging.info("Error occurring at threading {}, problem {}, index {} ".format(worker, sample["problem"], sample["index"]))
                
                #for type in ["ast", "cfg", "cdg", "ddg"]:
                #    with open(os.path.join(output_path, sample["index"],"{}.txt".format(type)), "r") as f:
                #        sample[type] = f.readlines()
                #self.data[index] = sample

        #return self.data
    def prepare(self):
        if (self.parallel == True):
            self.generate_path_based_multithreading()
        else:
            self.generate_path_based_singlethreading(self.data)

def chunkByProblem(seq, num):
    problems = list(set(sample["problem"] for sample in seq))
    avg = len(problems) / float(num)
    out = []
    last = 0.0

    while last < len(problems):
        out.append(problems[int(last):int(last + avg)])
        last += avg
    result = []
    for chunk_problem in out:
        result.append([sample for sample in seq if sample["problem"] in chunk_problem])
        
    return result