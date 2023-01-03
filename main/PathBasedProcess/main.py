from cpg_extractor import *
from tqdm import tqdm
import os
import threading
import logging 
import sys
from multiprocess import process
dataset_dir = "../dataset/"
output_dir = "./output/"

logging.disable(sys.maxsize)

def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def Joern(Jobs, num):
    
    for index in tqdm(range(len(Jobs))):
        folder = Jobs[index]
        for file in os.listdir(dataset_dir + folder):
            file_name = file.split(".")[0]
            if (os.path.exists(os.path.join("../cpg_8_2/",folder, file_name)) == False):
                try:
                    if (file.endswith(".cpp")):
                        input_path = dataset_dir + folder + '/' + file
                        output_path = output_dir + folder + '/'

                        if (os.path.exists(output_path) == False):
                            os.mkdir(output_path)

                        filename = file.split(".")[0]
                        
                        if (os.path.exists(os.path.join(output_path, filename)) == False):
                            os.mkdir(os.path.join(output_path, filename))

                        cnt = 0
                        for x in os.listdir(os.path.join(output_path, filename)):
                            if (x.endswith(".txt")): cnt+=1

                        if len(os.listdir(output_path)) - 1 < len(os.listdir(dataset_dir + folder)) and cnt != 4:
                            CPG_Extractor(input_path, output_path, filename).extract()
                except:
                    with open("logs.txt", "a") as f:
                        f.writelines(str(folder) + " " + str(file))

NumThread = 1
if __name__ == "__main__":
    
    JobList = list()
    for folder in os.listdir(dataset_dir):
        if (os.path.isdir(dataset_dir + folder)):
                for file in os.listdir(dataset_dir + folder):
                    if (file.endswith(".cpp")):
                        input_path = dataset_dir + folder + '/' + file
                        output_path = output_dir + folder + '/'

                        if (os.path.exists(output_path) == False):
                            os.mkdir(output_path)

                        filename = file.split(".")[0]
                        
                        if (os.path.exists(os.path.join(output_path, filename)) == False):
                            os.mkdir(os.path.join(output_path, filename))
                        cnt = 0
                        for x in os.listdir(os.path.join(output_path, filename)):
                            if (x.endswith(".txt")): cnt+=1
                        if len(os.listdir(output_path)) - 1 < len(os.listdir(dataset_dir + folder)) and cnt != 4:
                            JobList.append(folder)
    JobList = list(set(JobList))
    JobnList = chunk(JobList,NumThread)
    threads = [0] * NumThread
    for index in range(NumThread):
        threads[index] = threading.Thread(target = Joern, args = (JobnList[index], index,))
        
        threads[index].start()
    
    for index in range(NumThread):
        threads[index].join()
