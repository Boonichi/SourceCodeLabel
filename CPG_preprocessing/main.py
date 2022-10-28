from cpg_extractor import *
from tqdm import tqdm
import os
import threading

dataset_dir = "./dataset/"
output_dir = "./cpg_8_2/"

def Joern(Jobs):
    for index in range(len(Jobs)):
        folder = Jobs[index]
        print(index, '/', len(Jobs))
        for file in os.listdir(dataset_dir + folder):
            if (file.endswith(".cpp")):
                input_path = dataset_dir + folder + '/' + file
                output_path = output_dir + folder + '/'

                if (os.path.exists(output_path) == False):
                    os.mkdir(output_path)

                filename = file.split(".")[0]
                
                if (os.path.exists(os.path.join(output_path, filename)) == False):
                    os.mkdir(os.path.join(output_path, filename))

                if len(os.listdir(output_path)) - 1 < len(os.listdir(dataset_dir + folder)) and len(os.listdir(os.path.join(output_path, filename))) != 4:
                    CPG_Extractor(input_path, output_path, filename).extract()

NumThread = 2
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

                        if len(os.listdir(output_path)) - 1 < len(os.listdir(dataset_dir + folder)) and len(os.listdir(os.path.join(output_path, filename))) != 4:
                            JobList.append(folder)
    JobList = list(set(JobList))
    thread1 = threading.Thread(target = Joern, args = (JobList[:int(len(JobList) / 2)],)) 

    thread2 = threading.Thread(target = Joern, args = (JobList[int(len(JobList) / 2):],))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
