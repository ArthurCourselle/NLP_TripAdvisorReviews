from datasets import load_dataset
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import gc
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)

# Use CPU with explicit memory management
device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Configuration parameters
BATCH_SIZE = 16
EMBED_SIZE = 128  # Increased for word-level model
HIDDEN_SIZE = 256  # Increased for word-level model
MAX_SAMPLES = 25000  # Limit dataset size for faster training # MAX = 201295
MAX_SEQUENCE_LENGTH = 100  # Max number of words per review
EPOCHS = 20 # Increased epochs for better learning
MIN_WORD_FREQ = 3  # Minimum frequency for a word to be included in vocabulary
LEARNING_RATE = 0.001

class WordLevelReviewDataset(Dataset):
    def __init__(self, dataframe, vocab=None, max_seq_length=MAX_SEQUENCE_LENGTH):
        self.data = dataframe
        self.max_seq_length = max_seq_length
        
        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"
        
        if vocab is None:
            self.build_vocab()
        else:
            self.word2idx = vocab
            
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def build_vocab(self):
        # Initialize with special tokens
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.START_TOKEN: 2,
            self.END_TOKEN: 3
        }
        
        # Count word frequencies
        word_counts = Counter()
        for _, row in self.data.iterrows():
            # Clean and tokenize review
            words = self.tokenize(row['review'])
            word_counts.update(words)
        
        # Add words that appear frequently enough
        idx = 4  # Start after special tokens
        for word, count in word_counts.items():
            if count >= MIN_WORD_FREQ and word not in self.word2idx:
                self.word2idx[word] = idx
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)} words")

    def tokenize(self, text):
        """Clean and tokenize text into words"""
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        # Tokenize into words
        words = word_tokenize(text)
        return words

    def encode(self, text):
        """Convert text to sequence of word indices"""
        words = self.tokenize(text)
        # Limit sequence length and add start/end tokens
        words = [self.START_TOKEN] + words[:self.max_seq_length-2] + [self.END_TOKEN]
        return [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
    
    def decode(self, indices):
        """Convert sequence of indices back to text"""
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
        # Remove special tokens
        words = [word for word in words if word not in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]]
        return ' '.join(words)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        title_encoded = self.encode(row['title'])
        review_encoded = self.encode(row['review'])
        
        features = torch.tensor([
            row['overall'], row['cleanliness'], row['value'],
            row['location'], row['rooms'], row['sleep_quality']
        ], dtype=torch.float32) / 5.0  # Normalize to [0,1]
        
        return {
            'features': features,
            'title': torch.tensor(title_encoded, dtype=torch.long),
            'review': torch.tensor(review_encoded, dtype=torch.long)
        }


class WordLevelReviewGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.title_encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        self.fc_features = nn.Linear(6, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Increased capacity for word-level modeling
        self.lstm = nn.LSTM(
            embed_size + hidden_size * 2, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, title, review_in):
        # Encode title
        embedded_title = self.dropout(self.embed(title))
        _, (h_title, _) = self.title_encoder(embedded_title)
        h_title = h_title[-1].unsqueeze(1)
        
        # Encode features
        features_embedded = self.dropout(torch.relu(self.fc_features(features))).unsqueeze(1)
        
        # Create context
        context = torch.cat((h_title, features_embedded), dim=2)
        
        # Replicate context for each word in the review
        batch_size, seq_len = review_in.size()
        context = context.repeat(1, seq_len, 1)
        
        # Encode review input
        embedded_review = self.dropout(self.embed(review_in))
        
        # Concatenate review with context
        lstm_input = torch.cat((embedded_review, context), dim=2)
        
        # Process with LSTM
        output, _ = self.lstm(lstm_input)
        logits = self.fc_out(output)
        
        return logits

    def generate(self, features, title, vocab, max_length=50, temperature=0.7, top_k=40):
        """Generate text with better sampling strategies"""
        device = next(self.parameters()).device
        self.eval()
        
        # Get vocabulary mappings
        idx2word = {idx: word for word, idx in vocab.items()}
        start_token_idx = vocab.get("<START>", 2)
        end_token_idx = vocab.get("<END>", 3)
        
        with torch.no_grad():
            # Encode title
            embedded_title = self.embed(title.unsqueeze(0))
            _, (h_title, _) = self.title_encoder(embedded_title)
            h_title = h_title[-1].unsqueeze(1)
            
            # Encode features
            features_embedded = torch.relu(self.fc_features(features.unsqueeze(0))).unsqueeze(1)
            
            # Create context
            context = torch.cat((h_title, features_embedded), dim=2)
            
            # Start with the start token
            current_token = torch.tensor([[start_token_idx]], device=device)
            generated = []
            hidden = None
            
            # Generate sequence
            for _ in range(max_length):
                # Encode current token
                embedded_input = self.embed(current_token)
                
                # Concatenate with context
                lstm_input = torch.cat((embedded_input, context), dim=2)
                
                # Get predictions
                output, hidden = self.lstm(lstm_input, hidden)
                logits = self.fc_out(output.squeeze(1))
                
                # Apply temperature and top-k sampling
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                token_idx = next_token.item()
                
                # Stop if end token is generated
                if token_idx == end_token_idx:
                    break
                    
                generated.append(token_idx)
                current_token = next_token
            
            # Convert token indices to words
            return ' '.join([idx2word.get(idx, "<UNK>") for idx in generated])


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    batch_features = torch.stack([item['features'] for item in batch])
    batch_titles = nn.utils.rnn.pad_sequence([item['title'] for item in batch], batch_first=True, padding_value=0)
    batch_reviews = nn.utils.rnn.pad_sequence([item['review'] for item in batch], batch_first=True, padding_value=0)
    
    # Ensure contiguous memory
    batch_features = batch_features.contiguous()
    batch_titles = batch_titles.contiguous()
    batch_reviews = batch_reviews.contiguous()
    
    # Create input (all but last token) and target (all but first token) sequences
    reviews_input = batch_reviews[:, :-1].contiguous()
    reviews_target = batch_reviews[:, 1:].contiguous()
    
    return batch_features, batch_titles, reviews_input, reviews_target


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=0.001, clip_grad=1.0,
               device=device, checkpoint_dir='checkpoints'):
    """Train model with gradient clipping and learning rate scheduling"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    model = model.to(device)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for features, titles, inputs, targets in train_loader:
            features = features.to(device)
            titles = titles.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features, titles, inputs)
            
            # Reshape for loss calculation
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Print progress
            if batch_count % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}")
                # Free memory
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, titles, inputs, targets in val_loader:
                features = features.to(device)
                titles = titles.to(device)
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(features, titles, inputs)
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets.reshape(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()
                
                # Free memory
                del features, titles, inputs, targets, outputs, outputs_flat, targets_flat
                gc.collect()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    
    return train_losses, val_losses


def save_vocab(vocab, path):
    """Save vocabulary as JSON dictionary"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to {path}")


def load_vocab(path):
    """Load vocabulary from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"Vocabulary loaded from {path}")
    return vocab


def load_model(model_class, model_path, vocab_size, embed_size, hidden_size, device=device):
    """Load a pre-trained model"""
    model = model_class(vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def generate_reviews(model, vocab, titles, ratings_list, temperature=0.7, max_length=50):
    """Generate multiple reviews for comparison"""
    reviews = []
    
    for i, (title, ratings) in enumerate(zip(titles, ratings_list)):
        print(f"\nGenerating review {i+1}:")
        print(f"Title: {title}")
        print(f"Ratings: {ratings}")
        
        # Encode title
        title_encoded = [vocab.get(word, vocab.get("<UNK>", 1)) for word in word_tokenize(title.lower())]
        title_tensor = torch.tensor(title_encoded, dtype=torch.long).to(device)
        
        # Prepare ratings
        features = torch.tensor(ratings, dtype=torch.float32).to(device) / 5.0
        
        # Generate with different temperatures
        for temp in [0.5, 0.7, 1.0]:
            review = model.generate(features, title_tensor, vocab, max_length=50, temperature=temp)
            print(f"\nTemperature {temp}:")
            print(review)
            reviews.append((title, ratings, temp, review))
    
    return reviews


def main():
    # Check if checkpoints exist
    checkpoint_dir = 'checkpoints'
    model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    vocab_path = os.path.join(checkpoint_dir, 'word_vocab.json')
    
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        # Generation mode
        print("Model and vocabulary found. Generation mode activated.")
        
        # Load vocabulary
        vocab = load_vocab(vocab_path)
        vocab_size = len(vocab)
        
        # Load model
        model = load_model(WordLevelReviewGenerator, model_path, vocab_size, 
                          embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, device=device)
        
        # Example generation with different review titles and ratings
        titles = [
            "Beautiful hotel with amazing view",
            "Decent budget hotel in good location",
            "Disappointing stay, needs improvement"
        ]
        
        ratings_list = [
            [5.0, 4.5, 4.0, 5.0, 4.0, 4.5],  # High ratings
            [3.5, 3.0, 4.0, 4.5, 3.0, 3.5],  # Medium ratings
            [2.0, 2.5, 2.0, 3.5, 1.5, 2.0]   # Low ratings
        ]
        
        # Generate reviews with different settings
        generate_reviews(model, vocab, titles, ratings_list)
        
    else:
        # Training mode
        print("No model found. Training mode activated.")
        
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset("jniimi/tripadvisor-review-rating")
        raw_data = pd.DataFrame(dataset['train'])
        
        # Clean and prepare data
        df = raw_data.drop(columns=['stay_year', 'post_date', 'freq', 'lang'])
        df = df.dropna()
        df = df.drop_duplicates()
        
        # Check review lengths
        df['review_length'] = df['review'].str.len()
        print(f"Average review length: {df['review_length'].mean()} characters")
        
        # Filter super long reviews that might cause memory issues
        # df = df[df['review_length'] < 2000]
        
        # Limit dataset size for faster training
        df = df.sample(n=min(MAX_SAMPLES, len(df))).reset_index(drop=True)
        print(f"Dataset size after limiting: {len(df)} samples")
        
        # Split into train/val
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        
        # Create word-level datasets
        print("Creating word-level datasets...")
        train_dataset = WordLevelReviewDataset(train_df)
        val_dataset = WordLevelReviewDataset(val_df, vocab=train_dataset.word2idx)
        
        # Save vocabulary
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_vocab(train_dataset.word2idx, vocab_path)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        # Initialize word-level model
        vocab_size = len(train_dataset.word2idx)
        model = WordLevelReviewGenerator(vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, 
                                         num_layers=2, dropout=0.2)
        print(model)
        
        # Train model
        print("Starting training...")
        train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                              epochs=EPOCHS, lr=LEARNING_RATE, device=device, 
                                              checkpoint_dir=checkpoint_dir)
        
        print("Training completed!")
        
        # Example generation
        title = "Beautiful hotel with amazing view"
        ratings = [5.0, 4.5, 4.0, 5.0, 4.0, 4.5]
        
        # Encode title
        title_encoded = [train_dataset.word2idx.get(word, train_dataset.word2idx["<UNK>"]) 
                       for word in train_dataset.tokenize(title)]
        title_tensor = torch.tensor(title_encoded, dtype=torch.long).to(device)
        
        # Prepare ratings
        features = torch.tensor(ratings, dtype=torch.float32).to(device) / 5.0
        
        # Generate review
        print("\nGenerating review:")
        print(f"Title: {title}")
        print(f"Ratings: {ratings}")
        
        generated_review = model.generate(features, title_tensor, train_dataset.word2idx)
        print("\nGenerated review:")
        print(generated_review)


if __name__ == "__main__":
    main()
