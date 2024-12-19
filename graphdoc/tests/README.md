# Tests

## Setup

We assume that you have a `.env` file in the root of your directory, and that it appears as follows: 

```
OPENAI_API_KEY=<your api key>
```

## Running 

Some tests require external calls, such as to a model provider. In such cases, we utilize a `fire` flag in order to indicate external calls should occur. 

```bash
poetry run pytest --fire 
```