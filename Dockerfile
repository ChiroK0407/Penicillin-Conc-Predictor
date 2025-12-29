FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Set workdir
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "Home.py", "--server.port=7860", "--server.address=0.0.0.0"]
