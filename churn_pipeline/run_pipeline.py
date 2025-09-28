from data.fetch_embeddings import fetch_embeddings
from models.train_model import train
from models.evaluate_model import evaluate

def run():
    X, y = fetch_embeddings()
    model = train(X, y, version="latest")
    report = evaluate(model, X, y)
    print("Evaluation Report:", report)

if __name__ == "__main__":
    run()