import tensorflow as tf
import numpy as np

# Load and preprocess the dataset
with open("ptb.train.txt", "r") as f:
    text = f.read()

# Tokenize at the word level
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1

# Create input-output pairs
seq_length = 20
X, y = [], []
for i in range(seq_length, len(sequences)):
    X.append(sequences[i - seq_length:i])
    y.append(sequences[i])

X = np.array(X)
y = np.array(y)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=seq_length),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

# Train the model
model.fit(X, y, epochs=5, batch_size=128)

# Evaluate perplexity
def perplexity(y_true, y_pred):
    cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = tf.exp(tf.reduce_mean(cross_entropy))
    return perplexity

# Generate predictions for evaluation
y_pred = model.predict(X[:1000], verbose=0)
perp = perplexity(y[:1000], y_pred).numpy()
print(f"Perplexity on sample: {perp:.2f}")