from basemodel import BaseModel

'''
    There are three component of this architecture
    1. histNet
    2. detectNet
    3. removalNet
'''

class StcganGlobalHist(BaseModel):
    def __init__(self, *init_data, **kwargs):
        BaseModel.__init__(self, *init_data, **kwargs)
        print(self.__dict__)

        # mode: train

        if self.mode == 'train':
        # mode: test

        # load model
        if self.load_model:

    def train():
        pass


if __name__ == "__main__":
    sgh = StcganGlobalHist({'a': 2, 'b': 5}, path='hello')
    print(sgh.__dict__)
    pass