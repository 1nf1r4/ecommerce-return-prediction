# Justfile

# Default recipe: runs both frontend and backend together
default: run

# Run both services concurrently
run:
    echo "Starting frontend and backend..."
    (cd frontend && npm run dev) & (cd services && python main.py) & wait
