FROM nvcr.io/nvidia/pytorch:19.12-py3

RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pytorch-lightning==0.9.0
RUN pip install tokenizers
RUN pip install transformers