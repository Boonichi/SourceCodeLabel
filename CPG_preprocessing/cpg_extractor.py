#Import Library
from logging import raiseExceptions
import os
import shutil
import re
import psutil
import gc
import ray
from pathExtractor.extract_paths import *
import random

#Delete name of digraph in Joern File
def repl(m):
    return "digraph " + ' ' * (len(m.group(1)) - 2) + " {"

#Auto Garbage
def auto_garbage_collect(pct=15.0):
        if psutil.virtual_memory().percent >= pct:
            print("GARBAGE COLLECTED")
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

    for folder in os.listdir(output_dir):
        if folder.endswith("_output"):
            shutil.rmtree(os.path.join(output_dir, folder))
class CPG_Extractor():
    def __init__(self,input_path, output_path, filename, graph_name = "a", maxLength = 8, maxWidth = 2, 
                    maxTreeSize = 200, splitToken = 200, separator = "|", upSymbol = "^", downSymbol = "_", 
                    labelPlaceholder = "<SELF>", useParentheses = True, maxPathContext = 200):
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
        output_path = self.output_path + self.filename

        ast_output_path = output_path + "/ast_output/"
        cfg_output_path = output_path + "/cfg_output/"
        cdg_output_path = output_path + "/cdg_output/"
        ddg_output_path = output_path + "/ddg_output/"

        os.system("joern-parse " + input_path)

        if (os.path.exists(ast_output_path)): 
            shutil.rmtree(ast_output_path)
        os.system("joern-export --repr ast --out " + ast_output_path)

        if (os.path.exists(cfg_output_path)): 
            shutil.rmtree(cfg_output_path)
        os.system("joern-export --repr cfg --out " + cfg_output_path)

        if (os.path.exists(cdg_output_path)): 
            shutil.rmtree(cdg_output_path)
        os.system("joern-export --repr cdg --out " + cdg_output_path)

        if (os.path.exists(ddg_output_path)): 
            shutil.rmtree(ddg_output_path)
        os.system("joern-export --repr ddg --out " + ddg_output_path)

        #Edit joern file 
        self.edit_joern_file(ast_output_path)
        self.edit_joern_file(cfg_output_path)
        self.edit_joern_file(cdg_output_path)
        self.edit_joern_file(ddg_output_path)
        #Extract path from joern structure and transform them into Code2vec structure
        #AST paths:
        label = None
        ast_paths = []
        for ast_file in os.listdir(ast_output_path):
            if (ast_file.endswith(".dot")):
                auto_garbage_collect()
                label, ast_path = extract_ast_paths(os.path.join(ast_output_path + ast_file), self.graph_name, self.maxLength,
                                                                        self.maxWidth, self.maxTreeSize, self.splitToken, self.separator,
                                                                        self.upSymbol, self.downSymbol, self.labelPlaceholder,
                                                                        self.useParentheses)
                ast_paths.extend(ast_path)
        
        #CDG paths:
        source = "1000101"
        cfg_paths = []
        for cfg_file in os.listdir(cfg_output_path):
            if cfg_file.endswith(".dot"):
                cfg_paths.extend(extract_cfg_paths(os.path.join(cfg_output_path + cfg_file), self.graph_name,  source, self.splitToken, 
                    self.separator, self.upSymbol, self.downSymbol,
                    self.labelPlaceholder, self.useParentheses))        
        auto_garbage_collect()
        #CFG paths:
        cdg_paths = []
        for cdg_file in os.listdir(cdg_output_path):
            if cdg_file.endswith(".dot"):
                cdg_paths.extend(extract_cdg_paths(os.path.join(cdg_output_path, cdg_file), self.graph_name,
                    self.splitToken, self.separator, self.upSymbol, self.downSymbol,
                    self.labelPlaceholder,
                    self.useParentheses))
        auto_garbage_collect()
        #DDG paths:
        ddg_paths = []
        for ddg_file in os.listdir(ddg_output_path):
            if ddg_file.endswith(".dot"):
                try:
                    ddg_paths.extend(extract_ddg_paths(os.path.join(ddg_output_path, ddg_file), self.graph_name, source,
                                                       self.splitToken, self.separator, self.upSymbol, self.downSymbol,
                                                       self.labelPlaceholder,
                                                       self.useParentheses))
                except: pass

        #Set limit number of path
        ast_ouput_path = self.set_path_limit(ast_output_path)
        cdg_output_path = self.set_path_limit(cdg_output_path)
        cfg_output_path = self.set_path_limit(cfg_output_path)
        ddg_output_path = self.set_path_limit(ddg_output_path)

        #Extract into output_file

        write_output(output_path, "/ast.txt", ast_paths)
        write_output(output_path, "/cdg.txt", cdg_paths)
        write_output(output_path, "/cfg.txt", cfg_paths)
        write_output(output_path, "/ddg.txt", ddg_paths)
    
