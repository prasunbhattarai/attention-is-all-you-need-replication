import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformer import transformer_block
from datasets import load_dataset
from masks import src_mask, tgt_mask




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vocal_size =  1000
d_model = 128
max_len = 50
n_layer = 2
n_head = 4
dff = 512
dropout = 0.1
epochs = 2

dataset = load_dataset("opus_books", "en-fr")
tokenizer = AutoTokenizer.from_pretrained("t5_small")

def tokenize(dataset):
    input = tokenizer(dataset["translation"]["en"], padding = "max_length",trucation= True, max_length = max_len)
    target = tokenizer(dataset["translation"]["fr"], padding = "max_length",trucation= True, max_length = max_len)
    return {
        "input_ids": input["input_ids"],
        "attention_mask": input["attention_mask"],
        "labels": target["input_ids"]
    }

tokenized_dataset = dataset.map(tokenize, batched= True)

train_dataset = tokenized_dataset["train"].with_format("torch")

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

model = transformer_block(d_model,n_head,dff,dropout,vocal_size,max_len,n_layer)
model = model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameter(),lr = 1e-3)

for epoch in range (epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        src = batch["inputs_ids"].to(device)
        tgt = batch["labels"].to(device)

        tgt_input = tgt[:, :1]
        tgt_output = tgt[:, 1:]

        src_mask = src_mask(src).to(device)
        tgt_mask = tgt_mask(tgt).to(device)

        preds = model(src, tgt_input, src_mask, tgt_mask)

        loss = criterion(preds.view(-1, preds.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        print(f"Epoch {epoch+1:02d}, Loss: {total_loss/len(train_loader):.4f}")
