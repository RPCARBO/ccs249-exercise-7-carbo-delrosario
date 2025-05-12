import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load dataset from CSV
df = pd.read_csv("IMDB Dataset.csv")
df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})

# Preprocess text (same as traditional model)
def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c.isalnum() or c == " "])
    return text

df["cleaned_text"] = df["review"].apply(preprocess)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"], df["sentiment"], test_size=0.2, random_state=42
)

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 200  # Same sequence length as original
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

# Build LSTM model (same architecture)
model = Sequential([
    Embedding(input_dim=5000, output_dim=100, input_length=max_len),
    LSTM(128, dropout=0.5, recurrent_dropout=0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    X_train_pad, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Evaluate
y_pred_lstm = (model.predict(X_test_pad) > 0.5).astype("int32")

print("\nLSTM Model Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lstm):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lstm):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_lstm):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_lstm):.3f}")