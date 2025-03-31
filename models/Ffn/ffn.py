import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re

# -------------------------------
# Utility functions for text processing
# -------------------------------
def tokenize_and_encode(text, vocab):
    """Lowercases, tokenizes the text using a simple regex, and converts to token ids."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [vocab.get(word, vocab['<UNK>']) for word in tokens]

def build_vocab(texts, min_freq=2, max_vocab_size=10000):
    """Builds a vocabulary from a list of texts.
       Adds two special tokens: <PAD> for padding and <UNK> for unknown words.
    """
    counter = Counter()
    for text in texts:
        tokens = re.findall(r'\b\w+\b', text.lower())
        counter.update(tokens)
    # start with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.most_common(max_vocab_size):
        if freq < min_freq:
            break
        vocab[word] = len(vocab)
    return vocab

# -------------------------------
# Dataset class for reviews
# -------------------------------
class ReviewDataset(Dataset):
    def __init__(self, texts, ratings, vocab, max_length):
        self.texts = texts
        self.ratings = ratings
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        rating = self.ratings[idx]
        token_ids = tokenize_and_encode(text, self.vocab)
        # Truncate and pad the token list to fixed max_length
        token_ids = token_ids[:self.max_length]
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [self.vocab['<PAD>']] * (self.max_length - len(token_ids))
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        # Normalize rating to [0,1] (since ratings are 1-5, we use (rating-1)/4)
        norm_rating = (float(rating) - 1.0) / 4.0
        rating_tensor = torch.tensor([norm_rating], dtype=torch.float)
        return rating_tensor, token_ids

# -------------------------------
# Feedforward Network for "Language Modeling"
# -------------------------------
class FeedForwardLM(nn.Module):
    def __init__(self, input_dim, noise_dim, hidden_units, num_hidden_layers, vocab_size, max_length, activation, dropout_rate):
        """
        input_dim: Dimension of the rating feature (here 1)
        noise_dim: Dimension of injected noise for variability
        hidden_units: Number of units in each hidden layer
        num_hidden_layers: How many hidden layers to use
        vocab_size: Size of vocabulary (output dimension per token)
        max_length: Fixed length of generated review (number of tokens)
        activation: Activation function to use ('relu' or 'tanh')
        dropout_rate: Dropout probability applied after each hidden layer
        """
        super(FeedForwardLM, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        total_input_dim = input_dim + noise_dim
        
        layers = []
        in_features = total_input_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_units
        # The output layer projects to a vector that will be reshaped to (max_length, vocab_size)
        layers.append(nn.Linear(in_features, max_length * vocab_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, rating, noise):
        # Concatenate normalized rating (shape: [batch, 1]) with noise (shape: [batch, noise_dim])
        x = torch.cat([rating, noise], dim=1)
        out = self.model(x)  # shape: (batch, max_length * vocab_size)
        out = out.view(-1, self.max_length, self.vocab_size)
        return out

# -------------------------------
# Training and Evaluation Functions
# -------------------------------
def train_model(model, train_loader, val_loader, epochs, learning_rate, weight_decay, device, noise_dim):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD token (index 0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for ratings, targets in train_loader:
            ratings = ratings.to(device)
            targets = targets.to(device)
            batch_size = ratings.size(0)
            # Generate noise for each instance
            noise = torch.randn(batch_size, noise_dim).to(device)
            
            optimizer.zero_grad()
            outputs = model(ratings, noise)  # shape: (batch, max_length, vocab_size)
            # Reshape outputs and targets for computing cross-entropy loss
            outputs = outputs.view(-1, model.vocab_size)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} Training Loss: {avg_loss:.4f}")
        validate(model, val_loader, device, noise_dim)

def validate(model, val_loader, device, noise_dim):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for ratings, targets in val_loader:
            ratings = ratings.to(device)
            targets = targets.to(device)
            batch_size = ratings.size(0)
            noise = torch.randn(batch_size, noise_dim).to(device)
            outputs = model(ratings, noise)
            outputs = outputs.view(-1, model.vocab_size)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

# -------------------------------
# Generation function
# -------------------------------
def generate_review(model, rating, vocab, inv_vocab, device, noise_dim, temperature=1.0):
    """Generates a review given a rating (between 1 and 5).
       The rating is first normalized, noise is sampled, and then for each token position the model’s
       logits are converted to probabilities from which a token is sampled.
    """
    model.eval()
    with torch.no_grad():
        # Normalize the rating as done during training: (rating-1)/4
        norm_rating = (float(rating) - 1.0) / 4.0
        rating_tensor = torch.tensor([[norm_rating]], dtype=torch.float).to(device)
        noise = torch.randn(1, noise_dim).to(device)
        outputs = model(rating_tensor, noise)  # shape: (1, max_length, vocab_size)
        outputs = outputs.squeeze(0)  # shape: (max_length, vocab_size)
        tokens = []
        for i in range(outputs.size(0)):
            logits = outputs[i] / temperature
            probabilities = torch.softmax(logits, dim=0)
            token = torch.multinomial(probabilities, 1).item()
            # Optionally, stop if you hit a PAD token (or you can simply skip it)
            if token == vocab['<PAD>']:
                continue
            tokens.append(token)
        words = [inv_vocab.get(t, '<UNK>') for t in tokens]
        return " ".join(words)

# -------------------------------
# Main function
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Feedforward LM for review generation conditioned on rating")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"], help="Activation function")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--hidden_units", type=int, default=128, help="Number of units per hidden layer")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization)")
    parser.add_argument("--noise_dim", type=int, default=10, help="Dimension of noise vector for generation")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum token length for each review")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------
    # Load dataset
    # -------------------------------
    print("Loading dataset...")
    # The dataset file (a parquet file) is loaded from the given HF URL.
    df = pd.read_parquet("hf://datasets/jniimi/tripadvisor-review-rating/data/train-00000-of-00001.parquet")
    # We use the 'overall' column as the rating and the 'review' column as the text.
    texts = df['review'].astype(str).tolist()
    ratings = df['overall'].tolist()

    # -------------------------------
    # Build vocabulary
    # -------------------------------
    print("Building vocabulary...")
    vocab = build_vocab(texts, min_freq=2, max_vocab_size=10000)
    inv_vocab = {idx: word for word, idx in vocab.items()}
    print(f"Vocabulary size: {len(vocab)}")

    # -------------------------------
    # Split the dataset into training and validation sets
    # -------------------------------
    train_texts, val_texts, train_ratings, val_ratings = train_test_split(texts, ratings, test_size=0.2, random_state=42)
    train_dataset = ReviewDataset(train_texts, train_ratings, vocab, args.max_length)
    val_dataset = ReviewDataset(val_texts, val_ratings, vocab, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # -------------------------------
    # Instantiate the model
    # -------------------------------
    model = FeedForwardLM(
        input_dim=1,
        noise_dim=args.noise_dim,
        hidden_units=args.hidden_units,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=len(vocab),
        max_length=args.max_length,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
    )
    print(model)

    # -------------------------------
    # Train the model
    # -------------------------------
    train_model(model, train_loader, val_loader, args.epochs, args.learning_rate, args.weight_decay, device, args.noise_dim)

    # -------------------------------
    # Generation: interactively generate a review from a given rating
    # -------------------------------
    print("\nTraining complete. You can now generate a review by entering a rating (1-5).")
    while True:
        inp = input("Enter a rating (1-5) to generate a review (or 'quit' to exit): ")
        if inp.lower() == "quit":
            break
        try:
            rating_val = float(inp)
            if rating_val < 1 or rating_val > 5:
                print("Rating must be between 1 and 5")
                continue
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")
            continue

        generated_review = generate_review(model, rating_val, vocab, inv_vocab, device, args.noise_dim)
        print("\nGenerated review:")
        print(generated_review)
        print("-" * 40)

if __name__ == "__main__":
    main()
