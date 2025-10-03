# train_chatbot.py
import torch
import torch.nn as nn
import torch.optim as optim

# Training data (very small example)
data = [
    ("hello", "greet"),
    ("hi", "greet"),
    ("bye", "bye"),
    ("goodbye", "bye"),
    ("thanks", "thanks"),
    ("thank you", "thanks"),
]

labels = {"greet": 0, "bye": 1, "thanks": 2}

# Tokenize (bag of words style)
vocab = list(set(" ".join([d[0] for d in data]).split()))
word2idx = {w: i for i, w in enumerate(vocab)}

def vectorize(text):
    vec = torch.zeros(len(vocab))
    for w in text.split():
        if w in word2idx:
            vec[word2idx[w]] = 1
    return vec

X = torch.stack([vectorize(x) for x, y in data])
y = torch.tensor([labels[y] for x, y in data])

# Model
model = nn.Sequential(
    nn.Linear(len(vocab), 8),
    nn.ReLU(),
    nn.Linear(8, len(labels))
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(200):
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "chatbot.pt")
print("Model trained and saved!")
