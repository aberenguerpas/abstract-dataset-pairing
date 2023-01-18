from gensim.models import KeyedVectors
import os
import numpy as np
import pathlib

class Word2Vec:
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(os.path.join('../', pathlib.Path(__file__).parent.resolve()
,'models', 'GoogleNews-vectors-negative300.bin'),binary=True)
        self.dimensions = 300

    def getEmbedding(self, data):
        result =[]
        if len(data)>1:
            for element in data:
                list_tokens = [token for token in element.split(' ') if token in self.model]
                if list_tokens:
                    result.append(np.mean(self.model[list_tokens], axis=0))
        else:
            list_tokens = [token for token in data[0].split(' ') if token in self.model]

            if list_tokens:
                result.append(np.mean(self.model[list_tokens], axis=0))

        return result