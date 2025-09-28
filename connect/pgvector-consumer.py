from kafka import KafkaConsumer
from my_embeddings import embed
from my_vector_db import upsert_vector
import json

consumer = KafkaConsumer("mydatabase.public.crmreviews", 
                         "mydatabase.public.crmevents", 
                         bootstrap_servers="localhost:9092")

TABE_CONFIG = {
    "mydatabase.public.crmreviews": {
        "text_col": "reviews",
        "numeric_cols": ["stars"]
    },
    "mydatabase.public.crmevents": {
        "text_col": "consumer_complaint_narrative",
        "numeric_cols": ["stars"]
    }
}

for msg in consumer:
    # Parse the message (assuming JSON; if Avro, use Avro deserializer)
    event = json.loads(msg.value.decode("utf-8"))
    
    # Handle deletes
    if event["op"] == "d":
        delete_vector(event["before"]["id"])
        continue # skip the rest of the loop

    cfg = TABLE_CONFIG.get(msg.topic)
    if not cfg:
        continue # skip unknown topics

    # Generate embedding
    text = event["after"][cfg["text_col"]]
    numeric_features = {col: event["after"][col] for col in cfg["numeric_cols"]}
    vector = embed(text)
    metadata = {**event["after"], **numeric_features}
    upsert_vector(event["after"]["id"], vector, metadata=metadata)

    # Merge metadata with numeric features
    metadata = {**event["after"], **numeric_features}

    # Use the row's ID as the unique identifier
    upsert_vector(event["after"]["id"], vector, metadata=metadata)