FROM robd003/python3.9:latest

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN export FLASK_APP=pneumonia_detection
run flask run
CMD [ "python", "app.py" ]