import nltk
import re
from collections import defaultdict, Counter
import math

# tokenizer
# nltk.download('punkt')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load dataset
with open("ptb.train.txt", "r") as f:
    train_text = f.read()

# Clean text
train_text = clean_text(train_text)

# Tokenize
train_tokens = nltk.word_tokenize(train_text.lower())

# Build unigram counts
unigram_counts = Counter(train_tokens)

# Preprocessing: Remove rare words
min_count = 2  # Minimum frequency for a word to be included
train_tokens = [word for word in train_tokens if unigram_counts[word] >= min_count]

# Rebuild unigram, bigram, and trigram counts after preprocessing
unigram_counts = Counter(train_tokens)
bigram_counts = defaultdict(Counter)
trigram_counts = defaultdict(Counter)

for i in range(len(train_tokens) - 1):
    first, second = train_tokens[i], train_tokens[i + 1]
    bigram_counts[first][second] += 1
    if i < len(train_tokens) - 2:
        third = train_tokens[i + 2]
        trigram_counts[(first, second)][third] += 1

# Vocabulary size for smoothing
vocab = set(train_tokens)
V = len(vocab)

# Bigram probability with Laplace smoothing
def bigram_prob(w1, w2):
    return (bigram_counts[w1][w2] + 1) / (unigram_counts[w1] + V)

# Trigram probability with Laplace smoothing
def trigram_prob(w1, w2, w3):
    return (trigram_counts[(w1, w2)][w3] + 1) / (bigram_counts[w1][w2] + V)

# Calculate bigram perplexity
def calculate_bigram_perplexity(tokens):
    N = len(tokens) - 1
    log_prob = 0
    for i in range(N):
        w1, w2 = tokens[i], tokens[i + 1]
        prob = bigram_prob(w1, w2)
        log_prob += math.log(prob)
    return math.exp(-log_prob / N)

# Calculate trigram perplexity
def calculate_trigram_perplexity(tokens):
    N = len(tokens) - 2
    log_prob = 0
    for i in range(N):
        w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
        prob = trigram_prob(w1, w2, w3)
        log_prob += math.log(prob)
    return math.exp(-log_prob / N)

# Calculate perplexity on the training data
bigram_perplexity = calculate_bigram_perplexity(train_tokens)
trigram_perplexity = calculate_trigram_perplexity(train_tokens)

print(f"Bigram Perplexity: {bigram_perplexity:.2f}")
print(f"Trigram Perplexity: {trigram_perplexity:.2f}")
