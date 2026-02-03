# Import libraries
import os
import pickle
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  



class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        if "center" not in skipgram_df.columns or "context" not in skipgram_df.columns:
            raise ValueError(f"skipgram_df must have columns ['center','context'], got {list(skipgram_df.columns)}")
        self.centers = skipgram_df["center"].to_numpy(dtype=np.int64)
        self.contexts = skipgram_df["context"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return (
            torch.tensor(int(self.centers[idx]), dtype=torch.long),
            torch.tensor(int(self.contexts[idx]), dtype=torch.long),
        )


class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center_idx, context_idx):
        v = self.in_embed(center_idx)      
        u = self.out_embed(context_idx)
        return torch.sum(v * u, dim=1)     

    def get_embeddings(self):
        return self.in_embed.weight.detach()


def pick_device():
    return torch.device("cpu")


def build_negative_sampling_dist(counter, idx2word, vocab_size):
    counts = np.zeros(vocab_size, dtype=np.float64)

    if isinstance(idx2word, dict):
        for i in range(vocab_size):
            w = idx2word.get(i, idx2word.get(str(i), None))
            if w is None:
                continue
            counts[i] = float(counter.get(w, 0))
    else:
        for i, w in enumerate(idx2word):
            if i >= vocab_size:
                break
            counts[i] = float(counter.get(w, 0))

    if counts.sum() == 0:
        counts[:] = 1.0

    probs = counts ** 0.75
    probs = probs / probs.sum()
    return torch.tensor(probs, dtype=torch.float)


def sample_negatives(neg_dist, positive_context, num_negatives):
    B = positive_context.shape[0]
    neg = torch.multinomial(neg_dist, num_samples=B * num_negatives, replacement=True).view(B, num_negatives)

    mask = (neg == positive_context.unsqueeze(1))
    while mask.any():
        n_bad = int(mask.sum().item())
        neg[mask] = torch.multinomial(neg_dist, num_samples=n_bad, replacement=True)
        mask = (neg == positive_context.unsqueeze(1))

    return neg


def main():
    with open("processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    skipgram_df = data["skipgram_df"]
    word2idx = data["word2idx"]
    idx2word = data["idx2word"]
    counter = data["counter"]

    vocab_size = len(word2idx)

    device = pick_device()
    print("Using device:", device)

    neg_dist = build_negative_sampling_dist(counter, idx2word, vocab_size).to(device)

    dataset = SkipGramDataset(skipgram_df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = Word2Vec(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for center, context in pbar:
            center = center.to(device)
            context = context.to(device)

            pos_logits = model(center, context)
            loss_pos = criterion(pos_logits, torch.ones_like(pos_logits))

            neg_ctx = sample_negatives(neg_dist, context, NEGATIVE_SAMPLES)
            center_rep = center.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES).reshape(-1)
            neg_flat = neg_ctx.reshape(-1)

            neg_logits = model(center_rep, neg_flat)
            loss_neg = criterion(neg_logits, torch.zeros_like(neg_logits))

            loss = loss_pos + loss_neg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch}/{EPOCHS} - avg loss: {avg_loss:.4f}")

    embeddings = model.get_embeddings().cpu().numpy()
    payload = {"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}

    tmp_path = "word2vec_embeddings.pkl.tmp"
    out_path = "word2vec_embeddings.pkl"
    with open(tmp_path, "wb") as f: pickle.dump(payload, f)
    os.replace(tmp_path, out_path)

    print("Embeddings saved to word2vec_embeddings.pkl")
    print("Embeddings shape:", embeddings.shape)


if __name__ == "__main__":
    main()
