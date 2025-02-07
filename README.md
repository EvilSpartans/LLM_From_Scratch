# 🚀 Installation et Utilisation de l'Environnement

Ce guide vous permet de configurer et d'exécuter le **LLM**.

---

## 📦 Installation de l'Environnement Virtuel

```sh
python -m venv .venv
```

## 🏗️ Activation de l'Environnement Virtuel

- **Windows** :
  ```sh
  .venv\Scripts\Activate
  ```
- **Mac/Linux** :
  ```sh
  source .venv/bin/activate
  ```

---

## 🔧 Installation des Dépendances

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```sh
pip install -r requirements.txt
```

---

## 🚀 Exécution de chainlit

```sh
chainlit run chatbot.py
```

---

## 🎯 Remarques
- Assurez-vous d'avoir **Python 3.8+** installé sur votre machine.
- L'environnement virtuel permet d'éviter les conflits de versions entre les dépendances.
- Chainlit est un framework open-source permettant de créer des interfaces utilisateur.
- FastAPI est un framework léger et rapide pour créer des APIs modernes en Python.

---

🎉 **Tout est prêt !** 🚀
