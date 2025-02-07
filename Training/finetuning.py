import sys
import os
import json
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.classes import GPTModel

# Charger le modèle pré-entraîné
config = {
    "vocab_size": 50257,
    "context_length": 256,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(config).to(device)
model.load_state_dict(torch.load("./Models/gpt_fr.pth"))
model.train()

# Charger les instructions
dataset_path = os.path.join(os.path.dirname(__file__), "..", "Data", "instructions.json")
with open(dataset_path, "r", encoding="utf-8") as f:
    instructions = json.load(f)

# Initialisation du tokenizer
tokenizer = tiktoken.get_encoding("gpt2")


# 📌 **Nouvelle classe `InstructionDataset` adaptée à ton format**
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.encoded_texts = []

        for entry in data:
            question = entry["question"]
            answer = entry["answer"]
            formatted_text = f"Question : {question}\nRéponse : {answer}"

            # Tokenisation
            tokenized_text = torch.tensor(tokenizer.encode(formatted_text))
            self.encoded_texts.append(tokenized_text)

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.encoded_texts)


# 📌 **Nouvelle fonction de `collate` pour padding dynamique**
def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=50256)  # 50256 = <|endoftext|> pour GPT-2


# Création du dataset et DataLoader
train_dataset = InstructionDataset(instructions, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)

# Optimisation
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# Fine-tuning
for epoch in range(30):
    total_loss = 0
    for inputs in train_loader:
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Décalage des tokens pour correspondre aux cibles
        targets = inputs[:, 1:].contiguous()
        inputs = inputs[:, :-1].contiguous()

        loss = torch.nn.functional.cross_entropy(outputs[:, :-1, :].reshape(-1, config["vocab_size"]), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Fine-tuning Loss: {total_loss:.4f}")

# Sauvegarde du modèle fine-tuné
final_checkpoint_path = "Models/gpt_fr_finetuned.pth"
torch.save(model.state_dict(), final_checkpoint_path)
print("✅ Modèle fine-tuné sauvegardé !")
