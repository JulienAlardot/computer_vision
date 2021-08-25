FROM python:3.9

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN apt-get update
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip
EXPOSE 5000
COPY . .
RUN export FLASK_APP=pneumonia_detection
#RUN flask run
CMD [ "python", "test.py" ]