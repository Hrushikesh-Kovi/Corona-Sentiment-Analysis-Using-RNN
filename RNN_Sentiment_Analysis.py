import re
import unicodedata
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
warnings.filterwarnings("ignore")


print("Loading Data...")
df = pd.read_csv("Corona Tweet Sentiment Analysis/Corona_NLP_train.csv", encoding="latin1")
df = df[["OriginalTweet", "Sentiment"]]

labels = ["Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"]
df["label"] = df["Sentiment"].map({l: i for i, l in enumerate(labels)})

print("Cleaning Text...")
def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = re.sub(r"http\S+|@\w+", " ", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()

print("Tokenizing...")
lemmatizer = WordNetLemmatizer()

def get_tokens(text):
    tokens = word_tokenize(text)
    lemmas = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]
    return lemmas

df["cleaned_text"] = df["OriginalTweet"].apply(clean_text)
df["tokens"] = df["cleaned_text"].apply(get_tokens)

print("Splitting Data...")
train_df, test_df = train_test_split(
    df, random_state=42, test_size=0.2, stratify=df["label"]
)

word_counts = Counter()
for tokens in train_df["tokens"]:
    word_counts.update(tokens)

word_to_id = {
    "<pad>" : 0,
    "<unk>" : 1
}
for word in word_counts:
    word_to_id[word] = len(word_to_id)

print("Encoding Tokens...")
def encode_tokens(tokens):
    ids = [word_to_id.get(token, word_to_id["<unk>"]) for token in tokens]
    ids = ids[:50]
    return [0] * (50 - len(ids)) + ids

print("Making Loaders...")
def make_loader(data, shuffle):
    X = torch.LongTensor([encode_tokens(tokens) for tokens in data["tokens"]])
    y = torch.LongTensor(data["label"].values)
    return DataLoader(TensorDataset(X, y), shuffle=shuffle, batch_size=64)

train_loader = make_loader(train_df, True)
test_loader = make_loader(test_df, False)

print("Building Model...")
class RNNClassifier(nn.Module):

    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=100, padding_idx=0)
        self.rnn = nn.RNN(input_size=100, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        embeddings = self.embedding(x)
        _, hiddden = self.rnn(embeddings)
        return self.fc(self.dropout(hiddden[-1]))

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
model = RNNClassifier(len(word_to_id), num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training Model...")
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    epoch_loss = total_loss / len(train_loader)
    print(f"Epoch  :  {epoch + 1}/{epochs}  |  Loss  :  {epoch_loss:.4f}")

print("Evaluating Model...")
model.eval()
predictions = []
actual_vals = []

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        actual_vals.extend(y.cpu())
        logits = model(X)
        preds = torch.argmax(logits, dim=1).cpu()
        predictions.extend(preds)

print("Classification Report...")
print(classification_report(actual_vals, predictions))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(actual_vals, predictions), annot=True, fmt="d", cmap="mako")
plt.title("Confusion Matrix", fontweight=600, fontsize=24)
plt.show()

print("Program Execution Completed...")