from sentence_transformers import SentenceTransformer

class SentenceBert:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2') #device='cuda'
        self.tokenizer = None
        self.dimensions = 384

    def getEmbedding(self, data):
        return self.model.encode(data)
