import faiss
import pickle

def saveIndex(index, filename):
    faiss.write_index(index, filename)

def loadIndex(filename):
    f = open(filename, 'rb')

    reader = faiss.PyCallbackIOReader(f.read, 1234)
    reader = faiss.BufferedIOReader(reader, 1234)

    index = faiss.read_index(reader)

    return index

def createIndex(dimensions):
    index = faiss.IndexFlatIP(dimensions)
    index = faiss.IndexIDMap(index)
    return index

def saveInvertedIndex(d, path):
    pickled_file = open(path+'.pickle', 'wb')
    pickle.dump(d, pickled_file)

def loadInversedIndex(path):
    with open(path+'.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b