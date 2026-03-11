FROM --platform=linux/amd64 tensorflow/tensorflow:1.15.5-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get -qqq install --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/JiahuiYu/neuralgym.git /tmp/neuralgym && \
    pip install --no-cache-dir /tmp/neuralgym && \
    rm -rf /tmp/neuralgym

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install --no-cache-dir fastapi uvicorn[standard]

RUN adduser --disabled-password --gecos "" nonroot
USER nonroot

COPY --chown=nonroot:nonroot ./ /repo
WORKDIR /repo

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]       











