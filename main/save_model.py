from models.common import setup_model

from util import parser_config

import time


def save_model(args):
    model = setup_model(args, ContinueState=True)
    #save model
    file_path = args["SaveModelDir"]
    model.serialize(path = file_path)

if __name__ == "__main__":
    args = parser_config("config.json")
    save_model(args)