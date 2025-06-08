FROM nvidia/cuda:12.9.1-cudnn-runtime-ubi9

WORKDIR /app

RUN dnf install -y python3.12 python3.12-pip

COPY neural_net_cli.py neural_network.py requirements.txt /app/

RUN mkdir -p ./models && \
    python3 -m venv venv && \
    ./venv/bin/pip install --upgrade pip && \
    ./venv/bin/pip  install -r requirements.txt

CMD ["./venv/bin/python", "neural_net_cli.py", "optuna", "--dbs-path", "/mnt/dbs/", "--target-path", "/mnt/dbs"]