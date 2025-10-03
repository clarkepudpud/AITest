import torch
import torch.nn as nn
import json

# Load vocab and labels (from training script)
vocab = ["hello", "hi", "bye", "goodbye", "thanks", "thank", "you"]
word2idx = {w: i for i, w in enumerate(vocab)}
labels = {0: "greet", 1: "bye", 2: "thanks"}

def vectorize(text):
    vec = torch.zeros(len(vocab))
    for w in text.split():
        if w in word2idx:
            vec[word2idx[w]] = 1
    return vec

# Define model structure (must match training)
model = nn.Sequential(
    nn.Linear(len(vocab), 8),
    nn.ReLU(),
    nn.Linear(8, len(labels))
)

model.load_state_dict(torch.load("chatbot.pt", map_location="cpu"))
model.eval()

def handler(request):
    body = json.loads(request.body)
    text = body.get("message", "")

    with torch.no_grad():
        vec = vectorize(text)
        out = model(vec)
        pred = torch.argmax(out).item()
        intent = labels[pred]

    responses = {
        "greet": "Hello! How can I help you?",
        "bye": "Goodbye! Take care.",
        "thanks": "You're welcome!"
    }

    return {
        "statusCode": 200,
        "body": json.dumps({"reply": responses.get(intent, "I don't understand.")})
    }
