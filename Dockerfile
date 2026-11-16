FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_utils.py train.py predict.py ./
COPY archive/ archive/
COPY models/ models/

ENV HOST=0.0.0.0
ENV PORT=9696
EXPOSE 9696

CMD ["python", "predict.py"]