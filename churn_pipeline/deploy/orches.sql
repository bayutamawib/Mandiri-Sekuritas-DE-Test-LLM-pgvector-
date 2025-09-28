SELECT embedding, (cmetadata->>'label')::int FROM langchain_pg_embedding;
WHERE last_updated > NOW() - INTERVAL '30 days';