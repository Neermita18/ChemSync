import pandas as pd
import numpy as np
import requests
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import StringIO
def read():
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data, sep="\t").dropna()
    return df
url = 'https://www.cureffi.org/wp-content/uploads/2013/10/drugs.txt?raw=true'
response = requests.get(url, verify=False)
df=read()
df.dropna()
df["SMILES"] = df["smiles"]
df=df.drop(["smiles"], axis=1)
print(df)


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=512):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        inputs = self.tokenizer(smiles, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        inputs['labels'] = inputs.input_ids.detach().clone()
        
        # Create random mask
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != self.tokenizer.cls_token_id) * \
                   (inputs.input_ids != self.tokenizer.sep_token_id) * (inputs.input_ids != self.tokenizer.pad_token_id)
        
        selection = []
        for i in range(inputs.input_ids.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
        
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = self.tokenizer.mask_token_id
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['labels'] = inputs['labels'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)


        
        return inputs
    
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
smiles_list = df['SMILES'].tolist()
dataset = SMILESDataset(smiles_list, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print(len(dataset))
print(len(dataloader))


model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
optimizer = AdamW(model.parameters(), lr=1e-5)

model.train()
for epoch in range(3): 
    print(epoch)
    for batch in dataloader:
        # print(batch)
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Input IDs:", batch['input_ids'])
        print("Labels shape:", batch['labels'].shape)
        print("Labels:", batch['labels'])

        # Pass the batch to the model
        outputs = model(**batch)
        print("Outputs:", outputs)
        optimizer.zero_grad()
        inputs = {key: val.squeeze().to(model.device) for key, val in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss  
        loss.backward()
        optimizer.step()

model.save_pretrained("path_to_save_finetuned_model")