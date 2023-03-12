# Build step #1: build the React front end
FROM node:16-alpine as build-step
WORKDIR /app
COPY package.json ./
COPY ./client ./cleint


# Build step #2: build the API with the client as static files
FROM python:3.9
WORKDIR /app

RUN mkdir ./flask-server
COPY flask-server/requirements.txt flask-server/server.py flask-server/new-com-data.csv ./flask-server/
RUN pip install -r ./flask-server/requirements.txt
ENV FLASK_ENV production

EXPOSE 3000
WORKDIR /app/api
CMD ["gunicorn", "-b", ":3000", "api:app"]
