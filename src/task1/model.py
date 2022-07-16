import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class Task1Model(nn.Module):
    def __init__(self, params):
        super(Task1Model, self).__init__()
        self.bert_model = BertModel.from_pretrained(params['base_model'])
        bert_output_dim = self.bert_model.config.hidden_size
        self.num_labels = 2
        self.classification_layer = nn.Linear(bert_output_dim, self.num_labels)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizer.from_pretrained(params['base_model'])

        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and params['cuda'] else "cpu"
        )

        if (params['load_model_path'] is not None):
            self.load_model(params['load_model_path'])
    

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)


    def forward(self, 
        input_ids, 
        token_type_ids, 
        attention_mask, 
        labels=None
    ):
        outputs = self.bert_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        sentence_embeddings = outputs.pooler_output
        logits = self.classification_layer(sentence_embeddings)

        if (labels is None):
            return logits, None
        else:
            loss = self.criterion(logits, labels.view(-1))
            return logits, loss