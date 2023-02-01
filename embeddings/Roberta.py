from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class Roberta:

    def __init__(self):
       torch.cuda.empty_cache()
       self.model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
       self.dimensions = 1024

    def getEmbedding(self, data):
        res = []
        for d in data:
            d = d.split(" ")
            embeddings = self.model.encode(d)
            res.append(np.mean(embeddings, axis=0))        
        return res