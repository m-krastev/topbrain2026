# TopBrain 2026 TA36 submission (see inference.py). Build context is prepared by
# prepare.sh: build/nnUNet (UZH fork source) and build/*.py (our modules).
# cuda12.6 is required for the T4 GPU instances on Grand Challenge.
FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    nnUNet_n_proc_DA=1 \
    nnUNet_def_n_proc=1 \
    nnUNet_raw=/tmp \
    nnUNet_preprocessed=/tmp \
    nnUNet_results=/tmp

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

# venv on top of the image's torch (see the organisers' template)
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

# the fork pins CUDA 12.8 torch wheels; install it without dependencies
COPY --chown=user:user build/nnUNet /opt/app/nnUNet
RUN python -m pip install --no-cache-dir --no-color --no-deps /opt/app/nnUNet \
    && python -c "import torch, nnunetv2; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

COPY --chown=user:user build/predict_tta.py build/postprocess.py inference.py main.py torch_utilities.py /opt/app/

ENTRYPOINT ["python", "main.py"]
