import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class Task2TypeModel(nn.Module):
    def __init__(self, params):
        super(Task2TypeModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(params['base_model'])
        self.type_num = 3
        bert_output_dim = self.bert_model.config.hidden_size

        self.classification_layer = nn.Linear(bert_output_dim, self.type_num)
        self.classification_criterion = nn.BCEWithLogitsLoss()
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


    def predict(self,
        input_ids, 
        token_type_ids, 
        attention_mask, 
    ):
        outputs = self.bert_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
        )
        sentence_embeddings = outputs.pooler_output
        type_prediction = self.classification_layer(sentence_embeddings)
        predicted_types = (torch.sigmoid(type_prediction) > 0.5)

        return predicted_types


    def forward(self, 
        input_ids, 
        token_type_ids, 
        attention_mask, 
        labels,
    ):  
        outputs = self.bert_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
        )
        sentence_embeddings = outputs.pooler_output
        # sentence_embeddings = self.dropout(sentence_embeddings)
        type_prediction = self.classification_layer(sentence_embeddings)
        classify_loss = self.classification_criterion(type_prediction, labels.float())
        return classify_loss
        


class Task2TagModel(nn.Module):
    def __init__(self, params, tag_num):
        super(Task2TagModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(params['base_model'])
        bert_output_dim = self.bert_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        self.tag_num = tag_num
        self.tag_layer = nn.Linear(bert_output_dim, self.tag_num)  

        self.tag_criterion = nn.CrossEntropyLoss(reduction='mean')
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


    def predict(self,
        input_ids, 
        token_type_ids, 
        attention_mask, 
    ):
        outputs = self.bert_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
        )

        token_embeddings = outputs.last_hidden_state
        tag_predictions = self.tag_layer(token_embeddings)
        tag_predictions = torch.argmax(tag_predictions, dim=2)
        return tag_predictions


    def forward(self, 
        input_ids, 
        token_type_ids, 
        attention_mask, 
        tag_labels,
    ):  
        outputs = self.bert_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
        )
        token_embeddings = outputs.last_hidden_state
        tag_predictions = self.tag_layer(token_embeddings)
        tag_loss = self.tag_criterion(tag_predictions.view(-1, self.tag_num), tag_labels.view(-1))

        return tag_loss