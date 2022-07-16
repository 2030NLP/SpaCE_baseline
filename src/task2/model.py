import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class Task2Model(nn.Module):
    def __init__(self, params):
        super(Task2Model, self).__init__()
        self.bert_model = BertModel.from_pretrained(params['base_model'])
        bert_output_dim = self.bert_model.config.hidden_size

        self.type_num = 3
        self.classification_layer = nn.Linear(bert_output_dim, self.type_num)

        tag_nums = [2, 6, 3]
        self.tag_num = 12 # 2+6+3+1
        # self.tag_layers = [
        #     nn.Linear(bert_output_dim, tn+1) for tn in tag_nums
        # ]
        self.tag_layer = nn.Linear(bert_output_dim, self.tag_num)
        self.tag_mask = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ])

        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizer.from_pretrained(params['base_model'])

        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and params['cuda'] else "cpu"
        )

        self.tag_mask = self.tag_mask.to(self.device)
        
        if (params['load_model_path'] is not None):
            self.load_model(params['load_model_path'])
    

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)


    def predict(self,
        input_ids, 
        token_type_ids, 
        attention_mask, 
    ):
        outputs = self.bert_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        sentence_embeddings = outputs.pooler_output
        type_prediction = self.classification_layer(sentence_embeddings)
        predicted_types = torch.argmax(type_prediction, dim=1)

        token_embeddings = outputs.last_hidden_state
        tag_prediction = self.tag_layer(token_embeddings)
        tag_masks = self.tag_mask[predicted_types].view(-1, 1, self.tag_num)
        masked_prediction = tag_prediction * tag_masks

        return type_prediction, masked_prediction


    def forward(self, 
        input_ids, 
        token_type_ids, 
        attention_mask, 
        tag_labels,
        labels,
    ):
        outputs = self.bert_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        sentence_embeddings = outputs.pooler_output
        type_prediction = self.classification_layer(sentence_embeddings)
        classify_loss = self.criterion(type_prediction, labels)

        token_embeddings = outputs.last_hidden_state
        tag_prediction = self.tag_layer(token_embeddings)
        tag_masks = self.tag_mask[labels].view(-1, 1, self.tag_num)
        masked_prediction = tag_prediction * tag_masks
        tag_loss = self.criterion(masked_prediction.view(-1, self.tag_num), tag_labels.view(-1))

        return type_prediction, masked_prediction, classify_loss, tag_loss