from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class SciBert:
    def __init__(self):
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states = True)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dimensions = 768

    def getEmbedding(self, data):
        torch.cuda.empty_cache()
        res = []
        for d in data:
            d = d.split(" ")
            with torch.no_grad():
                tab = self.tokenizer(
                        d,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                ).to(self.device)

                self.model = self.model.to(self.device)
                output = self.model(**tab)
                res.append(np.mean([i[0].numpy() for i in output.last_hidden_state], axis=0))

        return res