FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt update && apt install -y tmux vim htop libgl1

RUN pip install matplotlib jupyterlab ipympl ipython scikit-learn scikit-fuzzy seaborn
RUN pip install torch torchvision segmentation-models-pytorch albumentations
RUN pip install opencv-contrib-python==4.5.5.62

WORKDIR /develop/results
WORKDIR /develop/data
WORKDIR /develop/code

