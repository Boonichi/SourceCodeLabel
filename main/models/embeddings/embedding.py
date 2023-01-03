from keras.preprocessing.text import Tokenizer as TK
from transformers import AutoTokenizer, TFAutoModel
from keras.utils import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

from util import dump_dataset
class BertTokenizer():
    def __init__(self, args):
        self.args = args
    def run(self):
        raise NotImplementedError

class Tokenizer():
    def __init__(self, args):
        self.args = args
        self.code_rep = ['ast', 'cfg', 'cdg', 'ddg']
        self.properties = ['source','path', 'value']
        self.params = self.args["embedding"]
        self.tokenizer_type = "Default"

    def prepare(self, dataset):
        self.tokenizer = dict()
        for rep in self.code_rep:
            for property in self.properties:
                self.tokenizer[rep + '_' + property] = TK(num_words = self.params[self.tokenizer_type]["num_words"], lower = True,)
                self.tokenizer[rep + '_' + property].fit_on_texts(dataset[rep + '_' + property].values)
        
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(dataset["tags"])

        dump_dataset("./storages/mlb.pickle", self.mlb, verbose = False)
        
    def tokenize(self, dataset):
        X = dict()
        for rep in self.code_rep:
            for property in self.properties:
                X[rep + '_' + property] = self.tokenizer[rep + '_' + property].texts_to_sequences(dataset[rep + "_" + property])
                X[rep + '_' + property] = pad_sequences(X[rep + '_' + property], maxlen = self.args["split"]["MAX_CONTEXT"][rep])
        
        Y = self.mlb.transform(dataset["tags"])
        return X, Y

    def run(self, dataset, case):
        if (case == "train"):
            self.prepare(dataset)
        return self.tokenize(dataset)
    
        
        