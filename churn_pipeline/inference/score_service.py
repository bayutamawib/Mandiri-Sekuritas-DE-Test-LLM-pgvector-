import os
import json
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
from confluent_kafka import Consumer, Producer
import tensorflow as tf
import yaml
import logging

# ------------------------------
# Configuration and Constants
# ------------------------------

DEFAULT_CONFIG_PATH = "src/churn_pipeline/config/model_config.yaml"

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "crm_embeddings")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "crm_embeddings_scored")
GROUP_ID = os.getenv("GROUP_ID", "churn_scorer_v1")

MODEL_PATH = os.getenv("MODEL_PATH", "models/keras_churn_vlatest") # Directory saved by model.save()
CONFIG_PATH = os.getenv("CONFIG_PATH", DEFAULT_CONFIG_PATH)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("score_service")

# ---------------------------
# Utilities
# ---------------------------

def load_config(path: str) -> Dict(str, Any):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def build_labeler(thresholds: Dict[str, float]):
    # thresholds: {high: 0.75, medium: 0.4}
    high = thresholds.get("high", 0.75)
    med = thresholds.get("medium", 0.4)

    def label_for(score: float) -> str:
        if score >= high:
            return "high"
        elif score >= med:
            return "medium"
        else:
            return "low"
        
        return label_for

def extract_features(
        message: Dict[str, Any],
        feature_names: List[str]
) ->    Tuple[np.ndarray, np.ndarray]:
        """
        Extract embedding vector and numeric feature vector from message.
        Respects the order defined in config["numeric_features"].
        Raises ValueError if missing or mismatched.
        """
        # Embedding
        embedding = message.get("embedding")
        if embedding is None or not isinstance(embedding, list):
            raise ValueError("Missing or invalid 'embedding' (expected list of floats)")
        emb_arr = np.array(embedding, dtype="float32")
        if emb_arr.ndim != 1:
            raise ValueError(f"'embedding must be 1D, got shape {emb_arr.shape}")

        # Features dict
        feats = message.get("features")
        if feats is None or not isinstance(feats, dict):
            raise ValueError(f"Missing or invalid 'features' (expected object).")

        # Extract in declared order
        numeric_values = []
        missing = []
        for name in feature_names:
            if name not in feats:
                missing.append(name)
                numeric_values.append(0.0) # fallback; still record missing for logs
            else:
                numeric_values.append(feats[name])
        
        if missing:
            logger.warning(f"Missing features {missing}; using 0.0 defaults for {message.get('contact_id')}")
        
        num_arr = np.array(numeric_values, dtype="float32")

        return emb_arr, num_arr

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------------------------
# Initialization
# ---------------------------

logger.info("Loading model config...")
config = load_config(CONFIG_PATH)
embedding_dim = int(config["embedding_dim"])
numeric_feature_names = list(config.get("numeric_features", []))
thresholds = dict(config.get("thresholds", {"high": 0.75, "medium": 0.4}))
model_version = str(config.get("model_version", "churn-v1.0"))

label_for = build_labeler(thresholds)

logger.info(f"Embedding dim: {embedding_dim}, numeric features: {numeric_feature_names}")
logger.info(f"Thresholds: {thresholds}, model_version: {model_version}")

logger.info(f"Loading Keras model from {MODEL_PATH}...")
model = tf.keras.models.load_odel(MODEL_PATH)
logger.info("Model loaded.")

consumer = Consumer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "group.id": GROUP_ID,
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False
})
producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

running = True
def _graceful_shutdown(sig, frame):
    global running
    logger.info(f"Received signal {sig}. Shutting down gracefully...")
    running = False
signal.signal(signal.SIGINT, _graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)

consumer.subscribe([INPUT_TOPIC])
logger.info(f"Subscribed to topic: {INPUT_TOPIC} -> producing to {OUTPUT_TOPIC}")

# ---------------------------
# Main Loop
# ---------------------------

def score_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run model inference for a single payload and return enriched event.
    """
    event_id = payload.get("event_id")
    contact_id = payload.get("contact_id")

    emb_vec, num_vec = extract_features(payload, numeric_feature_names)

    # Validate shapes vs config
    if emb_vec.shape[0] != embedding_dim:
        raise ValueError(f"Embedding dim mismatch: got {emb_vec.shape[0]}, expected {embedding_dim}")
    
    numeric_dim = len(numeric_feature_names)
    if num_vec.shape[0] != numeric_dim:
        raise ValueError(f"Numeric dim mismatch: got {num_vec.shape[0]}, expected {numeric_dim}")
    
    # Prepare batch dimension
    emb_batch = np.expand_dims(emb_vec, axis=0)
    num_batch = np.expand_dims(num_vec, axis=0)

    # Model expects named inputs
    preds = model.predict(
        {"embedding_input": emb_batch, "numeric_input": num_batch},
        verbose=0
    )
    score = float(preds[0][0]) # sigmoid output in [0,1]
    label = label_for(score)

    enriched = {
        "event_id": event_id,
        "contact_id": contact_id,
        "churn_score": score,
        "churn_label": label,
        "model_version": model_version,
        "scored_at": now_utc_iso()
    }
    return enriched
def main_loop():
    while running:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            logger.error(f"Kafka error: {msg.error()}")
            continue
        try:
            payload = json.loads(msg.value().decode("utf-8"))
        except Exception as e:
            logger.exception(f"failed to parse JSON: {e}")
            continue
        try:
            enriched = score_one
            out_batch = json.dumps(enriched, ensure_ascii=False).encode("utf-8")
            producer.produce(OUTPUT_TOPIC, out_bytes)
            producer.poll(0) # trigger delivery callbacks
            consumer.commit(msg)
            logger.info(f"Scored event_id={enriched['event_id']} contact_id={enriched['contact_id']} score={enriched['churn_score']:.4f}")
        except Exception as e:
            logger.exception(f"Scoring failed for event_id={payload.get('event_id')}: {e}")
            # Optionally send to a dead-letter topic; for now, commit to skip
            consumer.commit(msg)
        
    # Flush on shutdown
    producer.flush(5)
    consumer.close()
    logger.info("Shutdown complete.")

__name__ == "__main__":
 try:
    main_loop()
 except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    logger.exception(f"Fatal error: {e}")
 try:
    consumer.close()
 except Exception:
    pass
 sys.exit(1)