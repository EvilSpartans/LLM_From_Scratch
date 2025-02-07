import sys
import os
import torch
import tiktoken

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.classes import GPTModel, create_dataloader_v1

# Charger le dataset
dataset_path = os.path.join(os.path.dirname(__file__), "..", "Data", "dataset.txt")
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset_text = f.read()

# Paramètres d'entraînement
batch_size = 8
max_length = 256
stride = 128
num_epochs = 20
learning_rate = 5e-4

# Création du DataLoader
tokenizer = tiktoken.get_encoding("gpt2")
train_loader = create_dataloader_v1(dataset_text, batch_size=batch_size, max_length=max_length, stride=stride)

# Configuration du modèle
config = {
    "vocab_size": 50257,
    "context_length": max_length,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(config).to(device)

# Optimisation
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Entraînement
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, config["vocab_size"]), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Sauvegarde du modèle
final_checkpoint_path = "Models/gpt_fr.pth"
torch.save(model.state_dict(), final_checkpoint_path)
print("Modèle sauvegardé !")
