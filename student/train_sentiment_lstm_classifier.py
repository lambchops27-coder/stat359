# Train Sentiment Lstm Classifier

# Import libraries
import os
import re
import random
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import gensim.downloader as api

# Introduce seed for reproducibility
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Tokenization & Fasttext sequence encoding
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)?")

def simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())

def sentence_to_sequence(
    sentence: str,
    ft_model,
    max_len: int = 32,
    embedding_dim: int = 300,
) -> np.ndarray:
    # Convert sentence to float32 array via Fasttext vectors
    # Pads with 0s if shorter and truncates longer texts
    tokens = simple_tokenize(sentence)
    seq = np.zeros((max_len, embedding_dim), dtype = np.float32)

    # Fill in up to max_len vectors
    for i, tok in enumerate(tokens[:max_len]):
        if tok in ft_model.key_to_index:
            seq[i] = ft_model[tok]
    return seq

# Dataset
class PhrasebankSeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# Defining LSTM Model 
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 300,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_classes: int = 3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_lens =32, embedding_dim 300)
        output, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        # take final layer's hidden state -  both directions if bidirectional
        if self.lstm.bidirectional:
            # Concetenate final forward & backward state
            # previous layer forward = -2, bacward = -1
            h_final = torch.cat((h_n[-2], h_n[-1]), dim = 1)
            # (batch, 2*hidden_size)
        else:
            h_final = h_n[-1] # (batch, hidden_size)
        return self.fc(h_final)

# Metrics and Plots
@torch.no_grad()

def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0

    # loss computed outside training loop; return metric here

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)

        preds = torch.argmax(logits, dim = 1)
    
        all_true.append(yb.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average = 'macro')
    average_loss = total_loss / len(loader.dataset)
    return accuracy, macro_f1, average_loss, y_true, y_pred

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_true = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        
        preds = torch.argmax(logits, dim = 1)
        all_preds.append(preds.detach().cpu().numpy())
        all_true.append(yb.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_preds)

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average = 'macro')
    return accuracy, macro_f1, average_loss

        

def save_curves(history: dict, outdir: str):
    os.makedirs(outdir, exist_ok = True)

    # Plot loss curve
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(outdir, "lstm_loss_curve.jpeg"), bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(outdir, "lstm_accuracy_curve.png"), bbox_inches="tight")
    plt.close()

    # Macro F1 curve
    plt.figure()
    plt.plot(history["train_f1"], label="train")
    plt.plot(history["val_f1"], label="val")
    plt.title("Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.savefig(os.path.join(outdir, "lstm_macro_f1_curve.png"), bbox_inches="tight")
    plt.close()

def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, outdir: str):
    os.makedirs(outdir, exist_ok = True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    # Annotate
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha = 'center', va = 'center')
        
    plt.savefig(os.path.join(outdir, 'lstm_confusion_matrix.jpeg'), bbox_inches = 'tight')
    plt.close()

# Main
def main():
    outdir = 'outputs'
    os.makedirs(outdir, exist_ok = True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # Step 1 Load dataset
    print("******* Loading Dataset *******")
    ds = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    df = pd.DataFrame(ds["train"])
    print(df.head())
    print("\nLabel counts:\n", df["label"].value_counts())

    sentences = df["sentence"].values
    labels = df["label"].values
        
    # Step 2 Split Test, Validation & Test

    X_trainvali, X_test, y_trainvali, y_test = train_test_split(
        sentences,
        labels,
        test_size = 0.15,
        stratify = labels,
        random_state = 42,
    )
    X_train, X_vali, y_train, y_val = train_test_split(
        X_trainvali,
        y_trainvali,
        test_size = 0.15,
        stratify = y_trainvali,
        random_state = 42,
    )
    print(f"\nSplit sizes: train={len(X_train)}, val={len(X_vali)}, test={len(X_test)}")

    # Load fasttext
    print("******* Loading FastText *******")
    fasttext = api.load('fasttext-wiki-news-subwords-300')

    
    # Step 4 Precompute sequences
    print(" ******* Precomputing FastText sequences *******")
    max_len = 32
    emb_dim = 300
    X_train_seq = np.stack([sentence_to_sequence(s, fasttext, max_len, emb_dim) for s in X_train])
    X_val_seq   = np.stack([sentence_to_sequence(s, fasttext, max_len, emb_dim) for s in X_vali])
    X_test_seq  = np.stack([sentence_to_sequence(s, fasttext, max_len, emb_dim) for s in X_test])

    print("Train seq shape:", X_train_seq.shape)  # (N, 32, 300)

    # Data Loaders
    batch_size = 64
    train_loader = DataLoader(PhrasebankSeqDataset(X_train_seq, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(PhrasebankSeqDataset(X_val_seq, y_val), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(PhrasebankSeqDataset(X_test_seq, y_test), batch_size=batch_size, shuffle=False)

    # Class weight
    class_counts = np.bincount(y_train, minlength = 3).astype(np.float32)
    weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight = torch.tensor(weights, dtype=torch.float32, device=device))

    history = {
  "train_loss": [], "val_loss": [],
  "train_acc": [],  "val_acc": [],
  "train_f1": [],   "val_f1": [],
}

    best_val_f1 = -1.0
    best_path = os.path.join(outdir, "best_lstm.pt")

    
# Training the model
    model = LSTMClassifier(input_size = 300, hidden_size = 128, num_classes =3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    num_epochs = 30
    for ep in range(1, num_epochs + 1):
        train_acc, train_f1, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1, val_loss, _, _ = eval_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)

        print(f"Epoch {ep:02d} | "
      f"train acc {train_acc:.4f} f1 {train_f1:.4f} loss {train_loss:.4f} | "
      f"val acc {val_acc:.4f} f1 {val_f1:.4f} loss {val_loss:.4f}")

    # Save curves 
    save_curves(history, outdir)

    # Load BEST model after training
    model.load_state_dict(torch.load(best_path, map_location = device))

    te_acc, te_f1, te_loss, y_true, y_pred = eval_epoch(model, test_loader, criterion, device)
    print(f"Test acc {te_acc:.4f} | Test macroF1 {te_f1:.4f} | Test loss {te_loss:.4f}")
    save_confusion_matrix(y_true, y_pred, outdir)

if __name__ == "__main__":
    main()



 
