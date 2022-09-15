import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer


class Task3TriggerModel(nn.Module):
    def __init__(self, params, tag_num=4):
        super(Task3TriggerModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(params['base_model'])
        bert_output_dim = self.bert_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        self.tag_num = tag_num
        self.tag_layer = nn.Linear(bert_output_dim, self.tag_num)  

        self.tag_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.tokenizer = AutoTokenizer.from_pretrained(params['base_model'])

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


class Task3ElementModel(nn.Module):
    def __init__(self, params, tag_num=19):
        super(Task3ElementModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(params['base_model'])
        bert_output_dim = self.bert_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        self.tag_num = tag_num
        self.tag_layer = nn.Linear(bert_output_dim, self.tag_num)  
        self.tag_criterion = nn.CrossEntropyLoss(reduction='mean')

        self.factual_bert = BertModel.from_pretrained(params['base_model'])
        self.factual_layer = nn.Linear(bert_output_dim, 2)  
        self.factual_criterion = nn.CrossEntropyLoss(reduction='mean')

        self.tokenizer = AutoTokenizer.from_pretrained(params['base_model'])

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

        factual_outputs = self.factual_bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
        )
        sentence_embedding = factual_outputs.pooler_output
        fact_predictions = self.factual_layer(sentence_embedding)
        fact_predictions = torch.argmax(fact_predictions, dim=1)

        return tag_predictions, fact_predictions


    def forward(self, 
        input_ids, 
        token_type_ids, 
        attention_mask, 
        tag_labels,
        fact_labels,
    ):
        tag_outputs = self.bert_model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
        )
        token_embeddings = tag_outputs.last_hidden_state
        tag_predictions = self.tag_layer(token_embeddings)
        tag_loss = self.tag_criterion(tag_predictions.view(-1, self.tag_num), tag_labels.view(-1))

        factual_outputs = self.factual_bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
        )
        sentence_embedding = factual_outputs.pooler_output
        factual_predictions = self.factual_layer(sentence_embedding)
        fact_loss = self.factual_criterion(factual_predictions.view(-1, 2), fact_labels.view(-1))

        return tag_loss, fact_loss