from transformers import AutoTokenizer, AutoModel
import torch

class SciBert:

    def __init__(self):
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states = True)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dimensions = 768

    def getEmbedding(self, data):
        with torch.no_grad():
            tab = self.tokenizer(
                    data,
                    return_tensors="pt"
            ).to(self.device)

            print(tab)

            self.model = self.model.to(self.device)
            output = self.model(**tab)
            return [i[0] for i in output.last_hidden_state]