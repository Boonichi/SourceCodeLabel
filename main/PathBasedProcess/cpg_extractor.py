#Import Library
from logging import raiseExceptions, info
import os
import shutil
import re
import psutil
import gc
import ray
from PathBasedProcess.pathExtractor.extract_paths import *
import random

from util import syscmd_without_logging

from util import ensure_path

#Delete name of digraph in Joern File
def repl(m):
    return "digraph " + ' ' * (len(m.group(1)) - 2) + " {"

#Auto Garbage
def auto_garbage_collect(pct=15.0):
        if psutil.virtual_memory().percent >= pct:
            gc.collect()

def modify_path(path):
    path = re.sub(r"[<|)|(]","", path)
    return path

#Write token, path, value into file
def write_output(output_dir, namegraph, output):

    with open(output_dir + namegraph,"w") as output_file:
        for path in output:
            (token, path, value) = path
            path = modify_path(path)
            output_file.write(str(token) + ',' + str(path) + ',' + str(value) + "\n")

    #for folder in os.listdir(output_dir):
    #    if folder.endswith("_output"):
    #        shutil.rmtree(os.path.join(output_dir, folder))
class CPG_Extractor():
    def __init__(self,input_path, output_path, filename, graph_name = "a", maxLength = 8, maxWidth = 2, 
                    maxTreeSize = 200, splitToken = 200, separator = "|", upSymbol = "^", downSymbol = "_", 
                    labelPlaceholder = "<SELF>", useParentheses = True, maxPathContext = 200, WorkerIndex = None):
        #Intial parameters

        self.input_path = input_path
        self.output_path = output_path
        self.filename = filename
        self.graph_name = graph_name
        self.maxLength = maxLength
        self.maxWidth = maxWidth
        self.maxTreeSize = maxTreeSize
        self.splitToken= splitToken
        self.separator = separator
        self.upSymbol = upSymbol
        self.downSymbol = downSymbol
        self.labelPlaceholder = labelPlaceholder
        self.useParentheses = useParentheses
        self.maxPathContext = maxPathContext
        self.WorkerIndex = WorkerIndex

    #Modifying Path context
    def edit_joern_file(self, graph_dir):
        
        for dirpath, dirnames, filenames in os.walk(graph_dir):
            for filename in [f for f in filenames if f.endswith(".dot")]:
                with open(os.path.join(dirpath, filename), "r+") as f:
                    file_content = f.readlines()
                with open(os.path.join(dirpath, filename), "w") as f:
                    for index,content in enumerate(file_content):
                        if (index == 0):
                            result = re.sub('digraph(.*){', repl, content)
                        else:
                            result = ""
                            content = content.replace(" ","")
                            for word in range(len(content)):
                                if (content[word] == "=" and content[word - 1] == "l" and content[word - 2] == "e" and content[word - 3] == "b" and content[word - 4] == "a" and content[word + 1] != "\""): result+="=\""
                                elif word == len(content) - 2 and content[word] == "]" and content[word - 1] != "\"":
                                    result+="\"]"
                                else:
                                    result+=content[word]
                        f.write(result)
    def set_path_limit(self, paths):
        if (len(paths) > self.maxPathContext):
            paths = random.sample(paths, self.maxPathContext)
        return paths

    def extract(self):
        
        input_path = self.input_path
        output_path = os.path.join(self.output_path,self.filename)
        type_output_path = dict()
        for type in ['ast','cfg', 'cdg', 'ddg']:
            type_output_path[type] = output_path + '/{}_output/'.format(type)
            syscmd_without_logging("joern-parse {} --output {}.cpg.bin".format(input_path, "./storages/workspace/" + str(self.WorkerIndex)))
            syscmd_without_logging("joern-export {}.cpg.bin --repr {} --out ".format("./storages/workspace/" + str(self.WorkerIndex), type) + type_output_path[type])

            self.edit_joern_file(type_output_path[type])
        #Extract path from joern structure and transform them into Code2vec structure
        #AST paths:
        label = None
        paths = dict()
        paths["ast"] = []
        for ast_file in os.listdir(type_output_path["ast"]):
            if (ast_file.endswith(".dot")):
                auto_garbage_collect()
                label, ast_path = extract_ast_paths(os.path.join(type_output_path["ast"] + ast_file), self.graph_name, self.maxLength,
                                                                        self.maxWidth, self.maxTreeSize, self.splitToken, self.separator,
                                                                        self.upSymbol, self.downSymbol, self.labelPlaceholder,
                                                                        self.useParentheses)
                paths["ast"].extend(ast_path)
        
        #CDG paths:
        source = "1000101"
        paths["cfg"] = []
        for cfg_file in os.listdir(type_output_path["cfg"]):
            if cfg_file.endswith(".dot"):
                paths["cfg"].extend(extract_cfg_paths(os.path.join(type_output_path["cfg"] + cfg_file), self.graph_name,  source, self.splitToken, 
                    self.separator, self.upSymbol, self.downSymbol,
                    self.labelPlaceholder, self.useParentheses))        
        auto_garbage_collect()
        #CFG paths:
        paths["cdg"] = []
        for cdg_file in os.listdir(type_output_path["cdg"]):
            if cdg_file.endswith(".dot"):
                paths["cdg"].extend(extract_cdg_paths(os.path.join(type_output_path["cdg"], cdg_file), self.graph_name,
                    self.splitToken, self.separator, self.upSymbol, self.downSymbol,
                    self.labelPlaceholder,
                    self.useParentheses))
        auto_garbage_collect()
        #DDG paths:
        paths["ddg"] = []
        for ddg_file in os.listdir(type_output_path["ddg"]):
            if ddg_file.endswith(".dot"):
                try:
                    paths["ddg"].extend(extract_ddg_paths(os.path.join(type_output_path["ddg"], ddg_file), self.graph_name, source,
                                                       self.splitToken, self.separator, self.upSymbol, self.downSymbol,
                                                       self.labelPlaceholder,
                                                       self.useParentheses))
                except: pass

        #Set limit number of path
        #ast_ouput_path = self.set_path_limit(ast_output_path)
        #cdg_output_path = self.set_path_limit(cdg_output_path)
        #cfg_output_path = self.set_path_limit(cfg_output_path)
        #ddg_output_path = self.set_path_limit(ddg_output_path)

        #Extract into output_file
        for type in ['ast', 'cdg', 'cfg', 'ddg']:
            write_output(output_path, "/{}.txt".format(type), paths[type])
    