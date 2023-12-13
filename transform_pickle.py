import numpy
import pickle


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


read = open("data/weibo/validate_id.pickle", 'r')
data = pickle.load(StrToBytes(read), encoding='iso-8859-1')
with open('data/weibo/validate_id.pickle_1', 'wb') as f:
    f.write(pickle.dumps(data))
