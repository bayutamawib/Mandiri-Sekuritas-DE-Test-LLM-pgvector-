from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow

def evaluate(model, X_test, y_test, verbose=True):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log.metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    if verbose:
        print(f"✅ Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}