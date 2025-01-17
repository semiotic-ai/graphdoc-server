# Tests

## Setup

We assume that you have a `.env` file in the root of your directory, and that it appears as follows: 

```
OPENAI_API_KEY=<your api key>
```

## Running 

Some tests require external calls, such as to a model provider. In such cases, we utilize a `fire` flag as an indicator that we should both make and cache external calls. 

```bash
poetry run pytest --fire 
```

Alternatively, one can use the `dry-fire` flag to mock external calls with cached data. It is important to note, that we do not currently error handle for scenarios where no local data is cached. This will be handled by a `fixture` at some point. 

```bash
poetry run pytest --dry-fire
```