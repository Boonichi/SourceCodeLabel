from util import load_dataset

class FeatureParser():
    def __init__(self, args)
class PathBasedParser(FeatureParser):
    feature_kind = "PathBased"

    def __init__(self, args):
        self.args = args
        self.params = args["features"]["PathBased"]
    
    def extract_features(self, samples):
        X = {
            "start" : [],
            "path" : [],
            "end" : []
        }
        
        return {
            0: X["start"],
            1: X["path"],
            2: X["end"]
        }