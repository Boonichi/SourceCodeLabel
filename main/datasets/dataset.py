from util import load_dataset, dump_dataset

class Dataset(object):
    def __init__(self, args, data):
        self.args = args
        self.data = data

    def dump_source(self,path: str, code: str):
        """ Write solution to file """
        with open(path, "w") as f:
            f.write(code)
            
    def remove_non_ascii(self, code):
        if not code:
            return
        return ''.join([i if ord(i) < 128 else ' ' for i in code])
    
    def remove_apostrophe(self, code):
        result = ""
        for index, ch in enumerate(code):
            if ch == '\'':
                if code[index - 1].isdigit() and code[index + 1].isdigit():
                    continue
            result +=ch
        return result

    def remove_external_includes(self, code):
        exclude = ["#import", "using namespace", "#define", "import", "pragma", "typedef"]
        lines = []
        for line in code.split("\n"):
            skip = False
            for restricted in exclude:
                if restricted in line:
                    skip = True
                    break
            if skip:
                continue
            lines.append(line)
        return "\n".join(lines)

    def remove_unused_code(self, code):

        excluded = ["#include", "#pragma", "using namespace", "#import", "typedef"]

        res = []
        in_comment_block = False

        for line in code.split("\n"):

            line = line.strip()
            if len(line) == 0:
                continue

            if line.startswith("//"):
                if line.endswith("\\\\"):
                    in_comment_block = True
                continue
            elif in_comment_block and line.endswith("\\"):
                in_comment_block = True
                continue
            elif in_comment_block:
                in_comment_block = False
                continue

            for s in excluded:
                if line.startswith(s) and "*/" not in line:
                    break
            else:
                res.append(line)

        return "\n".join(res)
        
    def preprocess(self, code):
        pipeline = [
            self.remove_non_ascii,
            self.remove_apostrophe,
            self.remove_external_includes,
            self.remove_unused_code
        ]
        for func in pipeline:
            code = func(code)
        return code
    def prepare(self):
        for index,sample in enumerate(self.data):
            sample["source"] = self.preprocess(sample["source"])
            self.data[index] = sample
        return self.data
        
    def serialize(self, path = None):
        dump_dataset(path, self.data)
    