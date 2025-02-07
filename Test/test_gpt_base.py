import sys
import os
import torch
import tiktoken

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.classes import GPTModel, generate_text_simple

# Charger le modèle de base
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
model.eval()

# Initialisation du tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Entrée utilisateur
prompt = "L’automatisation des workflows"

# Tokenisation et génération
encoded = torch.tensor([tokenizer.encode(prompt)]).to(device)
generated_tokens = generate_text_simple(model, encoded, max_new_tokens=100, context_size=config["context_length"])
generated_text = tokenizer.decode(generated_tokens[0].tolist())

print("\n🔹 **Texte généré par le modèle de base :**")
print(generated_text)
