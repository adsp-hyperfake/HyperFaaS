FROM nvidia/cuda:12.9.1-cudnn-runtime-ubi9

WORKDIR /app

RUN dnf install -y python3.12

COPY . /app/

RUN pip install -r requirements.txt

CMD ["./venv/bin/python", "neural_net_cli.py", "optuna", "--dbs-path", "/mnt/dbs/", "--target-path", "/mnt/dbs"]