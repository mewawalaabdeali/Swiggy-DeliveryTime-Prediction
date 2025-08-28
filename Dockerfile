FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY requirements-dockers.txt ./

RUN pip install -r requirements-dockers.txt 

COPY app.py ./
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./scripts/data_clean_utils.py ./scripts/data_clean_utils.py  
COPY ./run_information.json ./

EXPOSE 8000

CMD ["python", "./app.py"]