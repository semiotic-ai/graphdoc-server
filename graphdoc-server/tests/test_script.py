# ../graphdoc_server/keys/api_key_config.json ("admin_key")

curl -X POST http://127.0.0.1:6000/inference \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY_HERE" \
  -d '{
    "database_schema": "type User { id: ID!, name: String, email: String }"
  }'

curl http://127.0.0.1:6000/model/version \
  -H "X-API-Key: YOUR_API_KEY_HERE"

curl -X POST http://127.0.0.1:6000/api-keys/generate \
  -H "X-API-Key: YOUR_ADMIN_KEY_HERE"

curl http://127.0.0.1:6000/api-keys/list \
  -H "X-API-Key: YOUR_ADMIN_KEY_HERE"