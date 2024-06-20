FROM --platform=$BUILDPLATFORM python:3.10 AS builder

WORKDIR /app

COPY requirements.txt /app
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY model.h5 ./model.h5
COPY static ./static
COPY templates ./templates
COPY app.py ./app.py


CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]








