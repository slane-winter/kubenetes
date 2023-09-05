FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt update && apt install -y tmux vim htop

RUN pip install scikit-image jupyterlab ipympl seaborn 
RUN pip install --no-dependencies qudida albumentations timm pretrainedmodels efficientnet_pytorch segmentation-models-pytorch

WORKDIR /develop/results/logs
WORKDIR /develop/data
WORKDIR /develop/code
