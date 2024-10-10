FROM python:3.10.12

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt --no-cache-dir 

COPY . .

EXPOSE 7860

CMD ["python3", "interface.py"]