import numpy as np
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load the dataset
ds = load_dataset("jniimi/tripadvisor-review-rating")
raw_data = pd.DataFrame(ds['train'])

# Define text and label columns
text = 'review'
label = 'overall'

# Drop unnecessary columns
df = raw_data.drop(columns=['stay_year', 'post_date', 'freq', 'lang'])

# Drop rows with missing data
df = df.dropna()

# Drop duplicates
df = df.drop_duplicates()

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Sample a very small fraction for quick testing
df = df.sample(frac=0.05, random_state=42)

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[label], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df[label], random_state=42)

# Tokenize and pad the text data
tokenizer = Tokenizer(num_words=1000)  # Reduced vocabulary size
tokenizer.fit_on_texts(train_df[text].values)

X_train = tokenizer.texts_to_sequences(train_df[text].values)
X_val = tokenizer.texts_to_sequences(val_df[text].values)
X_test = tokenizer.texts_to_sequences(test_df[text].values)

# Use same max length for all datasets
max_length = 50  # Short sequence length for quick testing
X_train = pad_sequences(X_train, maxlen=max_length)
X_val = pad_sequences(X_val, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Verify shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# Encode the labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[label].values)
y_val = label_encoder.transform(val_df[label].values)
y_test = label_encoder.transform(test_df[label].values)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Print shapes to verify
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

class SimpleRNN:
    def __init__(self, vocabulary_size, embedding_size, hidden_size, output_size, learning_rate=0.001):
        # Model dimensions
        self.vocab_size = vocabulary_size 
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize embedding matrix
        self.E = np.random.randn(embedding_size, vocabulary_size) * 0.01
        
        # Initialize RNN weights
        self.U = np.random.randn(hidden_size, embedding_size) * 0.01
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01
        self.V = np.random.randn(output_size, hidden_size) * 0.01
        
        # Initialize biases
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)

    def forward(self, inputs):
        """
        Forward pass for a single sequence
        
        Args:
            inputs: sequence of word indices of shape (sequence_length,)
            
        Returns:
            hidden_states: all hidden states
            y_pred: output prediction
        """
        seq_length = len(inputs)
        
        # Initialize arrays for forward pass
        # +1 for initial state h[0]
        hidden_states = np.zeros((seq_length + 1, self.hidden_size))
        
        # Process each word in the sequence
        for t in range(seq_length):
            # One-hot encode the input word index
            x_one_hot = np.zeros(self.vocab_size)
            if inputs[t] < self.vocab_size:
                x_one_hot[inputs[t]] = 1
            
            # Get word embedding
            embedding = self.E @ x_one_hot
            
            # Update hidden state
            hidden_states[t+1] = np.tanh(
                self.U @ embedding + 
                self.W @ hidden_states[t] + 
                self.bh
            )
        
        # Final prediction based on the last hidden state
        logits = self.V @ hidden_states[-1] + self.by
        y_pred = self._softmax(logits)
        
        return hidden_states, y_pred
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # For numerical stability
        return exp_x / np.sum(exp_x)
    
    def compute_loss(self, y_pred, y_true):
        """Compute cross entropy loss"""
        return -np.sum(y_true * np.log(y_pred + 1e-9))
    
    def backward(self, inputs, hidden_states, y_pred, y_true):
        """
        Backward pass for a single sequence
        
        Args:
            inputs: sequence of word indices
            hidden_states: hidden states from forward pass
            y_pred: predicted output
            y_true: true label
        """
        seq_length = len(inputs)
        
        # Initialize gradients
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        dE = np.zeros_like(self.E)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # Output gradient
        dy = y_pred - y_true  # Shape: (output_size,)
        dby = dy
        
        # Gradient for V
        dV = np.outer(dy, hidden_states[-1])
        
        # Initialize hidden state gradient
        dh_next = np.dot(self.V.T, dy)
        
        # Backpropagation through time
        for t in reversed(range(seq_length)):
            # Gradient through tanh
            dtanh = (1 - hidden_states[t+1]**2) * dh_next
            dbh += dtanh
            
            # One-hot encode the input word index
            x_one_hot = np.zeros(self.vocab_size)
            if inputs[t] < self.vocab_size:
                x_one_hot[inputs[t]] = 1
            
            # Get word embedding
            embedding = self.E @ x_one_hot
            
            # Gradients for U, W, and input embedding
            dU += np.outer(dtanh, embedding)
            dW += np.outer(dtanh, hidden_states[t])
            dE_t = np.dot(self.U.T, dtanh)
            
            # Accumulate embedding gradients
            if inputs[t] < self.vocab_size:
                dE[:, inputs[t]] += dE_t
            
            # Gradient for next timestep
            dh_next = np.dot(self.W.T, dtanh)
        
        # Clip gradients to prevent explosion
        for grad in [dU, dW, dV, dE, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)
        
        return dU, dW, dV, dE, dbh, dby
    
    def update_params(self, dU, dW, dV, dE, dbh, dby):
        """Update model parameters with gradients"""
        self.U -= self.learning_rate * dU
        self.W -= self.learning_rate * dW
        self.V -= self.learning_rate * dV
        self.E -= self.learning_rate * dE
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby
        
    def train(self, X_train, y_train, epochs=5, batch_size=32, verbose=True):
        """Train the model"""
        n_samples = len(X_train)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            processed = 0
            
            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                batch_loss = 0
                
                # Process each sample in the batch
                for j in range(len(batch_X)):
                    # Forward pass
                    hidden_states, y_pred = self.forward(batch_X[j])
                    
                    # Compute loss
                    loss = self.compute_loss(y_pred, batch_y[j])
                    batch_loss += loss
                    
                    # Backward pass
                    dU, dW, dV, dE, dbh, dby = self.backward(batch_X[j], hidden_states, y_pred, batch_y[j])
                    
                    # Update parameters
                    self.update_params(dU, dW, dV, dE, dbh, dby)
                
                # Track progress
                processed += len(batch_X)
                epoch_loss += batch_loss
                
                if verbose and processed % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Processed {processed}/{n_samples}, Loss: {batch_loss/len(batch_X):.4f}")
            
            # Compute average loss for the epoch
            avg_loss = epoch_loss / n_samples
            losses.append(avg_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        predictions = []
        
        for sample in X:
            _, y_pred = self.forward(sample)
            predictions.append(y_pred)
            
        return np.array(predictions)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        predictions = self.predict(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        accuracy = np.mean(pred_classes == true_classes)
        return accuracy

# Use a small vocab size for testing
vocab_size = 1000
embedding_size = 8
hidden_size = 16
output_size = y_train.shape[1]

print(f"Vocabulary size: {vocab_size}")
print(f"Embedding size: {embedding_size}")
print(f"Hidden size: {hidden_size}")
print(f"Output size: {output_size}")

# Create the model
rnn = SimpleRNN(
    vocabulary_size=vocab_size,
    embedding_size=embedding_size, 
    hidden_size=hidden_size,
    output_size=output_size,
    learning_rate=0.01
)

# Train with a small number of epochs and small batch size
losses = rnn.train(X_train, y_train, epochs=100, batch_size=16)

# Evaluate
accuracy = rnn.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")