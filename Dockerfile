FROM python

ARG MODEL_LOCATION

ENV model_location $MODEL_LOCATION

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

WORKDIR /app/prod

CMD ["fastapi","run","model.py","--port","8888"]