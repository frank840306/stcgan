import os
from fileio import *


class BaseModel():
    def __init__(self, *init_data, **kwargs):
        for dictionary in init_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.save_cfg()




    def save_cfg(self):
        cfg_json = os.path.join(self.path.log_dir, 'cfg.json')
        writeJson(self.__dict__, cfg_json)

    def save(self, name):
        pass
    def load(self):
        pass