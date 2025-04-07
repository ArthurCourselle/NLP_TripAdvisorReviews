import numpy as np
from datasets import load_dataset
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout


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
df = df.sample(frac=1).reset_index(drop=True)

# Sample the data to avoid a very large training set
df = df.sample(frac=0.005)

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[label], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df[label], random_state=42)

# Tokenize and pad the text data
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(train_df[text].values)

X_train = tokenizer.texts_to_sequences(train_df[text].values)
X_val = tokenizer.texts_to_sequences(val_df[text].values)
X_test = tokenizer.texts_to_sequences(test_df[text].values)

max_length = max(len(max(X_train, key=len)), 
                 len(max(X_val, key=len)), 
                 len(max(X_test, key=len)))
X_train = pad_sequences(X_train, maxlen=max_length)
X_val = pad_sequences(X_val, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Encode the labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[label].values)
y_val = label_encoder.transform(val_df[label].values)
y_test = label_encoder.transform(test_df[label].values)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)


# Define the RNN model
def create_rnn_model1(vocab_size, embedding_dim, max_length, output_dim):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    model.add(SimpleRNN(units=128, return_sequences=False))  # You can also try LSTM or GRU here
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=output_dim, activation='softmax'))
    return model

# Parameters
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
vocab_size = 5000  # Same as num_words in Tokenizer
embedding_dim = 50
max_length = 50
output_dim = y_train.shape[1]  # Number of classes


def create_rnn_model(vocab_size, embedding_dim, max_length, output_dim):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
    model.add(LSTM(units=256, return_sequences=True))  # Using LSTM instead of SimpleRNN
    model.add(Dropout(0.5))  # Adding dropout layer
    model.add(LSTM(units=128))  # Adding another LSTM layer
    model.add(Dropout(0.5))  # Adding dropout layer
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=output_dim, activation='softmax'))
    return model

# Create the model
rnn_model = create_rnn_model(vocab_size, embedding_dim, max_length, output_dim)

# Compile the model
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
rnn_model.summary()

# Train the model
history = rnn_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss, test_accuracy = rnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

