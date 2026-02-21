# STAT 359 LLMs
# Train Sentiment MLP Classifier

# Import libraries
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import datasets
import random
import matplotlib.pyplot as plt
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# Set random seed
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # if CUDA is available set seed
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Defining utility functions before used 
# Tokenization
def tokenize(text):
    text = text.lower() # convert text to lowercase
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

# Mean pooled vectors 
def sentence_to_vector(sentence, model):
    tokens = tokenize(sentence)
    vectors = []
    for token in tokens: # lookup fasttext vector for each token
        if token in model.key_to_index:
            vectors.append(model[token])
            # average fasttext vector - one sentence vector (300,)
    if len(vectors) == 0: 
        return np.zeros(300)
    return np.mean(vectors, axis=0) # if no tokens in Fasttext vocab, return 0


# Defining neural network using nn
class MLPClassifier(nn.Module):
    # Initialize parent Pytorch class 
    def __init__(self):
        super().__init__()
        # create fully connected layers between nodes
            # 300-dimension
            # output size: 250
        self.fc1 = nn.Linear(300, 250)
        # activation function ReLU - prevent MLP staying linear transformation
        self.relu = nn.ReLU()
        # Output = 3 -> 3 classes of sentiment: Negative (0), Neutral (1), Positive (2)
        self.fc2 = nn.Linear(250, 3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x) # apply ReLU activation
        x = self.fc2(x)
        return x # compute raw logit values

if __name__ == '__main__':
    model = MLPClassifier()
    x = torch.randn(64, 300)
    logits = model(x)
    print('logit shape:', logits.shape)

# Load and view Financial Phrasebank Dataset
print("\n========== Loading Dataset ==========")
from datasets import load_dataset

dataset = load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][:5])


phrasebank_data = pd.DataFrame(dataset['train'])
print(phrasebank_data.head())
print(phrasebank_data['label'].value_counts())

sentences = phrasebank_data['sentence'].values
labels = phrasebank_data['label'].values

# Splitting 
# Training + Validation vs 15% Test
X_trainvali, X_test, y_trainvali, y_test = train_test_split(
    sentences,
    labels,
    test_size = 0.15,
    stratify = labels,
    random_state = 42
)

# Train vs Validation (15% of Train&vali)
X_train, X_vali, y_train, y_vali = train_test_split(
    X_trainvali,
    y_trainvali,
    test_size = 0.15,
    stratify = y_trainvali,
    random_state = 42
)
    
print('Sizes:')
print('Train:', len(X_train))
print('Validation:', len(X_vali))
print('Test:', len(X_test))

# Sanity check: class balance preserved
print("Class dist.(counts): ")
print("Train: ", np.bincount(y_train))
print('Validation:', np.bincount(y_vali))
print('Test: ', np.bincount(y_test))

# Load fasttext
fasttext = api.load('fasttext-wiki-news-subwords-300')

print('Converting sentences to FastText mean-pooled vectors')
X_train_vector = np.vstack([sentence_to_vector(sentence, fasttext) for sentence in X_train
    ])

X_vali_vector = np.vstack([
    sentence_to_vector(sentence, fasttext) for sentence in X_vali
])

X_test_vector = np.vstack([
    sentence_to_vector(sentence, fasttext) for sentence in X_test
])

# Stack vectors and produce matrix
print("Shapes:")
print("Train:", X_train_vector.shape)
print("Val: ", X_vali_vector.shape)
print("Test: ", X_test_vector.shape)

os.makedirs('outputs', exist_ok = True)

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

device = get_device()
print('Device:', device)

class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 128

train_loader = DataLoader(SentimentDataset(X_train_vector, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SentimentDataset(X_vali_vector, y_vali),   batch_size=batch_size, shuffle=False)
test_loader = DataLoader(SentimentDataset(X_test_vector, y_test),   batch_size=batch_size, shuffle=False)

# Class weights from train
counts = np.bincount(y_train, minlength = 3).astype(np.float32)
weights = 1.0 / counts
weights = weights / weights.sum()
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

print("Train class counts:", counts)
print("Class weights:", weights)

# Model + Loss & Optimizer
model = MLPClassifier().to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 1e-2)

def run_epoch (model, loader, train = True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            logits = model(Xb)
            loss = criterion(logits, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * Xb.size(0)

            preds = torch.argmax(logits, dim = 1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        acc = (y_pred == y_true).mean()
        macro_f1 = f1_score(y_true, y_pred, average = 'macro')
        return avg_loss, acc, macro_f1

# Training Loop
num_epochs = 30
best_val_f1 = -1.0

train_loss_hist, val_loss_hist = [], []
train_acc_hist, val_acc_hist  = [], []
train_f1_hist, val_f1_hist   = [], []

for epoch in range(1, num_epochs + 1):
    tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, train = True)
    va_loss, va_acc, va_f1 = run_epoch(model, val_loader, train = False)

    if va_f1 > best_val_f1:
        best_val_f1 = va_f1
        torch.save(model.state_dict(), 'outputs/best_mlp_fasttext.pth')
        print(f"Save best checkpoint (val f1 = {best_val_f1:.3f})")

    print(f"Epoch {epoch:02d} | Val F1: {va_f1:.3f}")

    train_loss_hist.append(tr_loss); val_loss_hist.append(va_loss)
    train_acc_hist.append(tr_acc); val_acc_hist.append(va_acc)
    train_f1_hist.append(tr_f1); val_f1_hist.append(va_f1)

    print(f"Epoch {epoch:02d} | "
      f"train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
      f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}")

     
epochs = np.arange(1, num_epochs + 1)

# Plotting Function
def save_curve(y_train, y_vali, title, ylabel, filename):
    plt.figure()
    plt.plot(epochs, y_train, label = 'train')
    plt.plot(epochs, y_vali, label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", filename))
    plt.close()

save_curve(train_loss_hist, val_loss_hist, "Loss vs Epoch", "Loss", "mlp_loss_curve.png")
save_curve(train_acc_hist,  val_acc_hist,  "Accuracy vs Epoch", "Accuracy", "mlp_acc_curve.png")
save_curve(train_f1_hist,   val_f1_hist,   "Macro F1 vs Epoch", "Macro F1", "mlp_f1_curve.png")

print("Saved training curves to outputs/")

# Test evaluation using best checkpoint 
model.load_state_dict(torch.load("outputs/best_mlp_fasttext.pth", map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_labels)

test_acc = (y_pred == y_true).mean()
test_f1 = f1_score(y_true, y_pred, average="macro")
print(f"\nTEST Accuracy: {test_acc:.4f}")
print(f"TEST Macro F1: {test_f1:.4f}")

# Vectorization
confmatrix = confusion_matrix(y_true, y_pred)

plt.figure()
plt.imshow(confmatrix)
plt.title("Confusion Matrix (MLP)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1,2], ["neg(0)", "neu(1)", "pos(2)"], rotation=30, ha="right")
plt.yticks([0,1,2], ["neg(0)", "neu(1)", "pos(2)"])
for i in range(3):
    for j in range(3):
        plt.text(j, i, str(confmatrix[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig("outputs/mlp_confusion_matrix.png")
plt.close()

print("Saved confusion matrix to outputs/mlp_confusion_matrix.png")


# Test - checking if output is (300,)
vec = sentence_to_vector(phrasebank_data['sentence'][0], fasttext)
print("Vector shape:", vec.shape)

