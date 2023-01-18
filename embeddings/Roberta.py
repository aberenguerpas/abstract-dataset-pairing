from sentence_transformers import SentenceTransformer

class Roberta:

    def __init__(self):
       self.model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
       self.dimensions = 1024

    def getEmbedding(self, data):
       
        embeddings = self.model.encode(data)
        
        return embeddings