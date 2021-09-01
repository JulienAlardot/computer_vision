FROM python:3.8-slim-buster

LABEL maintainer="Julien Alardot <alardotj.pro@@gmail.com>"

WORKDIR /usr/src/app
RUN export FLASK_APP=pneumonia_detection
RUN export USE_NNPACK=0

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

EXPOSE 5000
COPY . .
#RUN curl -O ml_core/pneumodia_model.pt https://drive.google.com/file/d/1YZMN4UPTxFTBGhKAN-lSmpvsjHa0q90p/view?usp=sharing
CMD [ "python", "./app/app.py" ]