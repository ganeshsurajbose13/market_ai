# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose port (Render provides $PORT)
EXPOSE 10000

# Start FastAPI using uvicorn with Render's port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
