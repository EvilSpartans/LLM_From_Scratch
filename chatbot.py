import sys
import os
import torch
import tiktoken
import chainlit as cl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.classes import GPTModel, generate

# 📌 **Configuration du modèle**
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

# 📌 **Chargement du modèle fine-tuné**
model = GPTModel(config).to(device)
model.load_state_dict(torch.load("./Models/gpt_fr_finetuned.pth"))
model.eval()

# 📌 **Initialisation du tokenizer**
tokenizer = tiktoken.get_encoding("gpt2")

# 📌 **Interface utilisateur via Chainlit**
@cl.on_message
async def respond(message: cl.Message):
    user_input = message.content.strip()

    # Format du prompt
    prompt = f"Question : {user_input}\nRéponse :"

    # Tokenisation et génération
    encoded = torch.tensor([tokenizer.encode(prompt)]).to(device)
    generated_tokens = generate(model, encoded, max_new_tokens=100, context_size=config["context_length"], temperature=0.7, top_k=50)
    generated_text = tokenizer.decode(generated_tokens[0].tolist())

    # Nettoyage du texte généré
    cleaned_text = generated_text.split("\nRéponse :")[1] if "\nRéponse :" in generated_text else generated_text
    cleaned_text = cleaned_text.split("<|endoftext|>")[0].strip()

    # Répondre à l'utilisateur via Chainlit
    await cl.Message(content=cleaned_text).send()

# 📌 **Lancer Chainlit**
if __name__ == "__main__":
    cl.run(port=8000, host="0.0.0.0", debug=True)
