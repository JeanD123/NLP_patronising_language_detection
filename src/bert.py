import torch
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, DistilBertTokenizer
from transformers import RobertaTokenizer
from transformers import BertPreTrainedModel, BertModel
from transformers import DistilBertPreTrainedModel, DistilBertModel
from transformers import EarlyStoppingCallback
from transformers import TrainerCallback
from transformers import RobertaForSequenceClassification, RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

import pandas as pd
import numpy as np
import os

from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score

# just checking somethingdf
from datasets import Dataset

# my custom function
import utils
import importlib
from sklearn.model_selection import train_test_split

if not torch.cuda.is_available():
  print('WARNING: You may want to change the runtime to GPU for faster training!')
  DEVICE = 'cpu'
else:
  DEVICE = 'cuda:0'

class BERT_PCL(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config) 
        self.bert = BertModel(config)
        self.classification = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                                  torch.nn.Linear(config.hidden_size, 2))
        
        self.init_weights()

    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits_a = self.classification(outputs[1])
        return logits_a


class DistillBert_PCL(DistilBertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config) 
        self.bert = DistilBertModel(config)
        self.classification = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                                  torch.nn.Linear(config.hidden_size, 2))
        
        self.init_weights()

    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None):  # Add labels to the method signature
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        last_hidden_state = outputs[0]
        logits_a = self.classification(last_hidden_state[:, 0])
        return logits_a



class RobertaPCL(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config) 
        self.roberta = RobertaModel(config)
        self.classification = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                                  torch.nn.Linear(config.hidden_size, 2))
        self.init_weights()

    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.classification(outputs.pooler_output)
        
        if labels is not None:
            # Assuming labels are given
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits, labels)
            return SequenceClassifierOutput(loss=loss, logits=logits)
        
        return logits

class PCLDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, input_set):
        self.tokenizer = tokenizer
        self.encodings = self.tokenizer(list(input_set['text']), return_tensors='pt', padding=True, max_length=128, truncation=True)
        self.labels = list(input_set['labels'])

    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        
        item["labels"] = torch.tensor(self.labels[idx])
        return item



class Trainer_PCL(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')  # Assuming 'label' is a key in inputs
        # print(inputs)
        outputs = model(**inputs)
        # outputs = outputs.logits

        # TASK 1 - predicting whether or not the text contains PCL
        weight = torch.tensor([1.0, 2.0]).to(inputs["input_ids"].device)
        loss_task = nn.CrossEntropyLoss(weight=weight)
        # loss_task = nn.CrossEntropyLoss()
        
        # Convert labels to tensor
        labels_tensor = labels.clone().detach().to(torch.long).to(inputs["input_ids"].device)
        # print(outputs, labels_tensor)
        loss = loss_task(outputs.view(-1, 2), labels_tensor.view(-1))
        if return_outputs:
            return (loss, (loss, outputs))
        else:
            return loss



def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy":accuracy, "precision": precision, "recall": recall, "f1": f1}

def all_augmentations(model_type : str, inserted = False, subbed = False, back_translated = True, deleted = False, swapped = False):
    df_train, df_val, df_test = utils.load_train_and_val()
    df_train_pos = utils.load_augmented(inserted = inserted, subbed = subbed, back_translated = back_translated, 
                                        deleted = deleted, swapped = swapped)
    df_train = pd.concat([df_train[['text', 'bin_label']], df_train_pos], axis=0)
    df_train['labels'] = df_train['bin_label'].astype(int)
    df_val['labels'] = df_val['bin_label'].astype(int)
    df_test['labels'] = df_test['bin_label'].astype(int)
    if model_type.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(model_type)
    elif model_type.startswith('distil'):
        tokenizer = DistilBertTokenizer.from_pretrained(model_type)
    elif model_type.startswith('roberta'):
        tokenizer = RobertaTokenizer.from_pretrained(model_type)
    train_dataset = PCLDataset(tokenizer, df_train)
    val_dataset = PCLDataset(tokenizer, df_val)
    test_dataset = PCLDataset(tokenizer, df_test.fillna(''))
    return train_dataset, val_dataset, test_dataset


def main_PCL_v1(model : BERT_PCL, args : TrainingArguments, train_dataset : PCLDataset, 
                val_dataset : PCLDataset, model_output : str):
    
    trainer = Trainer_PCL(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(model_output)
    return trainer



def predict_pcl(input, attention_mask, model):
    model.eval()
    # encodings = tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=128)

    output = model(input_ids=input, attention_mask=attention_mask)
    preds = torch.max(output, 1)
    return {'prediction': preds[1], 'confidence': preds[0]}


def evaluate(model, data_loader):
    total_count = 0
    correct_count = 0
    preds = []
    tot_labels = []

    with torch.no_grad():
        for data in tqdm(data_loader):
            labels = {}
            labels['label'] = data['labels']
            text = data['input_ids']
            attention = data['attention_mask']
            pred = predict_pcl(text, attention, model)

            preds.append(pred['prediction'].tolist())
            tot_labels.append(labels['label'].tolist())


    report = classification_report(tot_labels, preds, target_names=['No PCL', 'PCL'], output_dict=True)
    return report