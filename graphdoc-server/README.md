```bash
# From the parent directory
docker compose -f graphdoc-server/docker-compose.yml --profile dev up --build

# From the parent directory
docker compose -f graphdoc-server/docker-compose.yml --profile prod up --build
```

```bash
# From the root directory
docker compose -f docker-compose.yml --profile dev up --build

# From the parent directory
docker compose -f docker-compose.yml --profile prod up --build
```