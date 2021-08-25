FROM python:3.9

LABEL maintainer="Julien Alardot <alardotj.pro@@gmail.com>"

WORKDIR /usr/src/app

RUN apt-get update -y

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip
EXPOSE 5000
COPY . .
RUN export FLASK_APP=pneumonia_detection
CMD [ "python", "./app.py" ]