from datasets.dataset import Dataset

import os
from util import load_dataset, dump_dataset, run_system_command
from pathlib import Path

from tqdm import tqdm

class SourceDataset(Dataset):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tokenizer = self._ensure_tokenizer_exists()
        self.savedDir = "./storages/cpgDataset/"
    def _ensure_tokenizer_exists(self):

        tokenizer_dir  ="./tokenizer" +  "/src"

        tokenizer_exe = "tokenizer"

        tokenizer_path = tokenizer_dir + "/" + tokenizer_exe

        if not os.path.exists(tokenizer_path):
            current_path = Path.cwd()
            os.chdir(tokenizer_dir)
            run_system_command("{} *.cpp *.h -o {}".format(self.compiler, tokenizer_path))
            os.chdir(current_path)

        return tokenizer_path

    def split_source_tokens(self, code,source_path,tokens_path):
        self.dump_source(source_path, code)

        tokenizer_cmd = "{} -t c {}".format(self.tokenizer, source_path)
        with open(tokens_path, "w") as f:
            rc = run_system_command(tokenizer_cmd,
                                    stdout=f,
                                    stderr=f,
                                    verbose=False)
            if rc:
                raise Exception("Failure occured during tokenization!")

        with open(tokens_path, "r") as f:
            tokens = []
            for line in f:
                if "//" not in line and "/*" not in line:
                    tokens.append(line.strip())
                elif "EOF encountered" in line:
                    raise Exception("Failure occured during tokenizer!")
        return tokens
    def prepare(self):         
        for index, sample in tqdm(enumerate(self.data)):
            check = False

            output_path = os.path.join(self.savedDir, sample["problem"])
            
            for type in ['tokens']:
                if type not in sample.keys(): 
                    if (os.path.exists(os.path.join(output_path, sample["index"], "{}.pickle".format(type))) == False):
                        check = True
                        break
                
            if (check):
                tokens = self.split_source_tokens(sample["source"], "./storages/workspace/temp.cpp", "./storages/workspace/solution.tokens")
                dump_dataset(os.path.join(output_path,sample["index"], "tokens.pickle"), tokens, verbose = False)
            
                #for type in ["tokens"]:
                #    with open(os.path.join(output_path, sample["index"],"{}.txt".format(type)), "r") as f:
                #        sample[type] = f.readlines()
                #self.data[index] = sample

        #return self.data