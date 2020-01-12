FROM ubuntu:18.04

RUN apt-get update && \
  apt-get install -y software-properties-common 


RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv cmake
RUN apt-get install -y git

# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel
RUN mkdir workspace

COPY . /workspace

WORKDIR /workspace

RUN python3.6 -m pip install --upgrade -r requirements.txt
RUN apt update && apt install -y libsm6 libxext6 libxrender1

#convert the Keras model to ONNX 
RUN python3.6 convert.py

ENTRYPOINT ["python3.6", "server.py"]
