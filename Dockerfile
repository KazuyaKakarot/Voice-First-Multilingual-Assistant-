FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# API on 8000. For the Streamlit UI instead, override the command:
#   docker run ... scheme-assistant streamlit run ui/streamlit_app.py --server.address 0.0.0.0
EXPOSE 8000 8501
CMD ["python", "main.py", "serve"]
