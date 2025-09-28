FROM confluentinc/cp-kafka-connect:8.0.0

# Debezium Postgres connector
RUN confluent-hub install --no-prompt debezium/debezium-connector-postgresql:3.1.2

# Snowflake Kafka sink connector
RUN confluent-hub install --no-prompt snowflakeinc/snowflake-kafka-connector:3.2.4

# Tell Kafka Connect where to find them
ENV CONNECT_PLUGIN_PATH="/usr/share/confluent-hub-components,/usr/share/java"