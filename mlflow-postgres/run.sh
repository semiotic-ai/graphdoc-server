#!/bin/bash

# Check if an environment file is provided
if [ -z "$2" ]; then
    echo "Usage: $0 {start|stop} <env_file>"
    exit 1
fi

ENV_FILE=$2

# Load environment variables from the specified .env file
export $(grep -v '^#' "$ENV_FILE" | xargs)

function start() {
    echo "Starting PostgreSQL server with environment file $ENV_FILE..."
    (docker-compose up -d)
}

function stop() {
    echo "Stopping PostgreSQL server with environment file $ENV_FILE..."
    (docker-compose down)
}

function test_postgres() {
    start
    echo "Testing PostgreSQL server..."
    docker exec -it postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, name VARCHAR(50));"
    docker exec -it postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "INSERT INTO test_table (name) VALUES ('Test Name');"
    docker exec -it postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT * FROM test_table;"
    stop
}

# Check the command line argument
if [ "$1" == "start" ]; then
    start
elif [ "$1" == "stop" ]; then
    stop
elif [ "$1" == "test" ]; then   
    test_postgres
else
    echo "Usage: $0 {start|stop|test} <env_file>"
    exit 1
fi 