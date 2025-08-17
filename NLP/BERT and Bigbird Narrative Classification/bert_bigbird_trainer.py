###  This script is for TRAINING a BERTxBigbird model.  It uses BERT for smaller narratives and BigBird for longer narratives.  This is all used to pepare the model for inference.


import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler, random_split
from transformers import BertTokenizer, BertModel, BertConfig, BigBirdTokenizer, BigBirdModel, AutoTokenizer, AutoModelForSequenceClassification
import chardet
import traceback

#uses a GPU if available, otherwise uses the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Imports the data
file_path = 'data/fake_narratives.csv'
with open(file_path, 'rb') as file:
    result = chardet.detect(file.read())
    print(result)
dln = pd.read_csv(file_path,  encoding=result['encoding'])
dln = dln.dropna() #drops all the empty values

# This column "list" is a mult-hot encoded vector created by mushing together 8 columns of dummy variables in the data set that equal 1 if an incident that fits into that respective "aspect of concern" occurs the following day
dln['list'] = dln[dln.columns[5:]].values.tolist()
nlp_dln = dln[['narrative', 'list']].copy()

# Randomly sample n log entries. Intended for debugging.
# nlp_dln = nlp_dln.sample(n=1000, random_state=42)

###################################################################################################################################################################
### Here we tokenize the data and split the dataset into data for BERT and data for BIGBIRD, then create a custom dataset class that turns all this narrative businiess into stuff that the machines can read, then they pass through the model. 
#The model learns from the training dataset outputs its guesses on aspects of concern from the Validation Dataset.  

# Initialize tokenizers and pre-trained models

# I would like to use ModernBERT someday but I'm going to get one thing working at a time 
# model_id = "answerdotai/ModernBERT-base"
# bert_tokenizer = AutoTokenizer.from_pretrained(model_id)
# bert_model = AutoModelForSequenceClassification.from_pretrained(model_id, return_dict=True)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
bigbird_tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
bigbird_model = BigBirdModel.from_pretrained('google/bigbird-roberta-base', return_dict=False)

# Set a length threshold for model selection, the rest are parameters for the dataloader
length_threshold = 512
#Now that we have handled the max length, here are the rest of the training parameters
max_len_bert = 512
max_len_bigbird = 1536
train_batch_size = 64
valid_batch_size = 64
epochs = 1
learning_rate = 1e-05

# Split the dataset into BERT data and BigBird data
def split_dataset_with_tokenization(dataset, bert_tokenizer, bigbird_tokenizer, max_len_bert, max_len_bigbird):
    bert_data = []
    bigbird_data = []

    for _, row in dataset.iterrows():
        text = row['narrative']
        targets = row['list']

        # Tokenize and decide which model to use
        tokenized_text = bert_tokenizer.tokenize(text)

        if len(tokenized_text) <= max_len_bert:
            inputs = bert_tokenizer.encode_plus(
                text,
                max_length=max_len_bert,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
            )
            bert_data.append({
                "ids": inputs["input_ids"],
                "mask": inputs["attention_mask"],
                "token_type_ids": inputs["token_type_ids"],
                "targets": targets,
            })
        else:
            inputs = bigbird_tokenizer.encode_plus(
                text,
                max_length=max_len_bigbird,
                padding="max_length",
                truncation=True,
                return_token_type_ids=True,
            )
            bigbird_data.append({
                "ids": inputs["input_ids"],
                "mask": inputs["attention_mask"],
                "token_type_ids": inputs["token_type_ids"],
                "targets": targets,
            })

    # Convert to DataFrames for further processing

    return bert_data, bigbird_data

bert_data, bigbird_data = split_dataset_with_tokenization(
    nlp_dln,
    bert_tokenizer,
    bigbird_tokenizer,
    max_len_bert,
    max_len_bigbird
)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.indexes = list(range(len(data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return {
            'ids': torch.tensor(item['ids'], dtype=torch.long),
            'mask': torch.tensor(item['mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(item['token_type_ids'], dtype=torch.long),
            'targets': torch.tensor(item['targets'], dtype=torch.float),
            # 'index': self.indexes[index]  # Return the index itself; we need this for the weighted batch sampling
            # 'index': torch.tensor(item['index'], dtype=torch.long)
        }

#Split the data again into training and testing sets, one for each
train_size = 0.8

bert_dataset = CustomDataset(bert_data)
bigbird_dataset = CustomDataset(bigbird_data)

def split_dataset(dataset, val_fraction=0.2):
    data_len = len(dataset)
    val_size = int(data_len * val_fraction)
    train_size = data_len - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    return train_data, val_data

bert_train_dataset, bert_test_dataset = split_dataset(bert_dataset)
bigbird_train_dataset, bigbird_test_dataset = split_dataset(bigbird_dataset)

print("FULL Dataset: {}".format(nlp_dln.shape))
print("BERT TRAIN Dataset: {}".format(len(bert_train_dataset)))
print("BERT TEST Dataset: {}".format(len(bert_test_dataset)))
print("BigBird TRAIN Dataset: {}".format(len(bigbird_train_dataset)))
print("BigBird TEST Dataset: {}".format(len(bigbird_test_dataset)))

# Initialize the dataset class on the split datasets

###############################################################################################################################################################################
#This part of the code is bonus weighting to ensure that we oversample the positive case.  It is not essential to perform the training task but to make the model more accurate
# Create weights for each example based on its classification
# Calculate class weights based on the dataset

class_counts = np.sum(np.array(nlp_dln['list'].tolist()), axis=0) 
class_weights = np.clip(1.0 / (class_counts + 1e-6), 0, 1e3)
#class_weights = 1.0 / (class_counts + 1e-6)
  # Avoid division by zero
print("Class distribution in training data:")
print(class_counts)

sample_weights = []
for target in nlp_dln['list']:
    weights = [class_weights[i] for i, value in enumerate(target) if value > 0]
    
    # if len(weights) == 0:
    #     print(f"Warning: No positive classes for target {target}")  # Check for empty cases
    
    avg_weight = np.mean(weights) if weights else 0  # Avoid NaN if no positive classes
    sample_weights.append(avg_weight)
sample_weights = torch.tensor(sample_weights, dtype=torch.float)

if len(bert_dataset) > 0:
    bert_train_weights = sample_weights[bert_dataset.indexes]  # Ensure weights align with training data
    bert_sampler = WeightedRandomSampler(
        weights=bert_train_weights,
        num_samples=len(bert_dataset),  # Match the training data size
        replacement=True
    )
if len(bigbird_dataset) > 0:
    bigbird_train_weights = sample_weights[bigbird_dataset.indexes]  # Ensure weights align with training data
    bigbird_sampler = WeightedRandomSampler(
        weights=bigbird_train_weights,
        num_samples=len(bigbird_dataset),  # Match the training data size
        replacement=True
    )
###############################################################################################################################################################################
#initialize the dataloader, which feeds the data into the model

#this is basic random sampling
bert_train_loader = DataLoader(bert_train_dataset, batch_size=train_batch_size, sampler=RandomSampler(bert_train_dataset))
bert_test_loader = DataLoader(bert_test_dataset, batch_size=valid_batch_size, sampler=SequentialSampler(bert_test_dataset))
bigbird_train_loader = DataLoader(bigbird_train_dataset, batch_size=train_batch_size, sampler=RandomSampler(bigbird_train_dataset))
bigbird_test_loader = DataLoader(bigbird_test_dataset, batch_size=valid_batch_size, sampler=SequentialSampler(bigbird_test_dataset))

# this is custom sampling aiming to oversample the positive case so that we train the model has a chance to see more postive cases
# bert_train_loader = DataLoader(bert_train_dataset, batch_size=train_batch_size, sampler=bert_sampler)
# bert_test_loader = DataLoader(bert_test_dataset, batch_size=valid_batch_size, shuffle=False)
# bigbird_train_loader = DataLoader(bigbird_train_dataset, batch_size=train_batch_size, sampler=bigbird_sampler)
# bigbird_test_loader = DataLoader(bigbird_test_dataset, batch_size=valid_batch_size, shuffle=False)

#Creating the customized model
class DualEncoderClassifier(torch.nn.Module):
    def __init__(self):
        super(DualEncoderClassifier, self).__init__()

        #Initialize BERT and BigBird models
        self.bert = bert_model
        self.bigbird = bigbird_model
        #Dropout layer to avoid overfitting
        self.dropout = torch.nn.Dropout(0.3)
        #Output classification layer
        #This assumes hidden size of 768 for both BERT and BigBird
        self.classifier = torch.nn.Linear(768, 8)

    def forward(self, ids, mask, token_type_ids):
        #determines which model to use based on the input length
        if ids.size(1) <= 512:
            #Use BERT for the shorter sequences
           _,output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        else:
            #Use  BigBird if the sequence length is longer than 512
            _,output = self.bigbird(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        pooled_output = output
        #Pass through dropout layer
        output2 = self.dropout(pooled_output)

        #Pass through the classification layer
        outputs = self.classifier(output2)

        return outputs

#calls the model and moves to the appropriate device
model = DualEncoderClassifier()
model.to(device)


###################################################################################################################################################################
###LOSS###

# Convert class weights to a tensor
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Use weights in BCEWithLogitsLoss
#criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

# Define the Loss Function
criterion = torch.nn.BCEWithLogitsLoss()

def loss_fn(outputs, targets):
    return criterion(outputs, targets)

optimizer = torch.optim.Adam(params = model.parameters(), lr=learning_rate)

###################################################################################################################################################################

# Fine tuning the model
def train(epoch, train_loaders):
    model.train() #Sets the model to training mode
    total_loss = 0 #tracking loss across the epoch
    print(f"Starting epoch {epoch}...")

        #loads data into the device
    for loader in train_loaders:
        if len(list(loader)) == 0:
            print(f"Skipping empty loader.")
            continue

    for step, data in enumerate(train_loaders, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()

        # Prints the loss every 100 steps
        if step %100==0:
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss

train_loaders = [bert_train_loader, bigbird_train_loader]

print(f"Starting training for {epochs} epoch(s)...")
for epoch in range(epochs):
    for loader in train_loaders:
        train(epoch, loader)

###################################################################################################################################################################

#Validating the model
#During the validation stage we pass the unseen data(Testing Dataset) to the model. This step determines how good the model performs on the unseen data.

def validate(epoch, val_loaders):
    model.eval()
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for loader in val_loaders:
            for _, data in enumerate(loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)

                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().numpy().tolist())

    return np.array(fin_outputs), np.array(fin_targets)
    
val_loaders = [bert_test_loader, bigbird_test_loader]

for epoch in range(epochs):
    outputs, targets = validate(epoch, val_loaders)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(classification_report(targets, outputs >= 0.5, zero_division=0))


#save the model artifacts
torch.save(model.state_dict(), "PATH")
print("Model saved!")
