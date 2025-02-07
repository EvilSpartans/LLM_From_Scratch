import sys
import os
import torch
import tiktoken

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.classes import GPTModel, generate_text_simple, generate

# Charger le modèle fine-tuné
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
model.load_state_dict(torch.load("./Models/gpt_fr_finetuned.pth"))
model.eval()

# Initialisation du tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Entrée utilisateur (question)
question = "C'est quoi la programmation ?"

# Format spécifique basé sur ton dataset
prompt = f"Question : {question}\nRéponse :"

# Tokenisation et génération
encoded = torch.tensor([tokenizer.encode(prompt)]).to(device)
# generated_tokens = generate_text_simple(model, encoded, max_new_tokens=100, context_size=config["context_length"])
generated_tokens = generate(model, encoded, max_new_tokens=100, context_size=config["context_length"], temperature=0.7, top_k=50)

generated_text = tokenizer.decode(generated_tokens[0].tolist())

# Nettoyage pour enlever les éventuelles balises inutiles
cleaned_text = generated_text.split("\nRéponse :")[1] if "\nRéponse :" in generated_text else generated_text
cleaned_text = cleaned_text.split("<|endoftext|>")[0].strip()

print("\n🔹 **Réponse générée par le modèle fine-tuné :**")
print(cleaned_text)
