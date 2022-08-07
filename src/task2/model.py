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
        self.dropout = nn.Dropout(0.1)

        self.tag_nums = [3, 7, 4]
        self.tag_layers = nn.ModuleList([
            nn.Linear(bert_output_dim, tn) for tn in self.tag_nums 
        ])

        self.classification_criterion = nn.BCEWithLogitsLoss()
        self.tag_criterion = nn.CrossEntropyLoss(reduction='none')
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

        token_embeddings = outputs.last_hidden_state
        tag_predictions = [layer(token_embeddings) for layer in self.tag_layers]
        tag_predictions = torch.stack([torch.argmax(x, dim=2) for x in tag_predictions], dim=1) # batch_size*3*seq_len
        return predicted_types, tag_predictions


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
            attention_mask=attention_mask,
        )
        sentence_embeddings = outputs.pooler_output
        # sentence_embeddings = self.dropout(sentence_embeddings)
        type_prediction = self.classification_layer(sentence_embeddings)
        classify_loss = self.classification_criterion(type_prediction, labels.float())

        token_embeddings = outputs.last_hidden_state
        # token_embeddings = self.dropout(token_embeddings)
        tag_predictions = [layer(token_embeddings) for layer in self.tag_layers]
        
        tag_labels = torch.transpose(tag_labels, 0, 1) # batch_size*seq_len*type_num
        tag_losses = [
            self.tag_criterion(tag_predictions[i].view(-1, self.tag_nums[i]), tag_labels[i].reshape(-1)) for i in range(self.type_num)
        ]
        tag_losses = [
            x.reshape(-1, self.params['seq_max_length']) for x in tag_losses
        ]
        tag_losses = torch.stack(tag_losses, dim=2) # batch_size*seq_len*type_num
        tag_losses = tag_losses.mean(dim=1) # batch_size*type_num
        tag_loss = torch.mean(tag_losses * labels)

        return type_prediction, tag_predictions, classify_loss, tag_loss