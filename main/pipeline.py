from util import load_dataset

class SourcePipeline():
    def __init__(self, args):
        self.args = args

    def run(self,samples):
        X, X_meta = 