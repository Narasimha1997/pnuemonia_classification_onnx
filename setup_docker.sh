#!/bin/bash

mkdir tfwheel

#download the suitable tensorflow release for docker build

wget https://files.pythonhosted.org/packages/85/d4/c0cd1057b331bc38b65478302114194bd8e1b9c2bbc06e300935c0e93d90/tensorflow-2.1.0-cp36-cp36m-manylinux2010_x86_64.whl  -P ./tfwheel/

docker build . -t pnuemonia_onnx:latest
