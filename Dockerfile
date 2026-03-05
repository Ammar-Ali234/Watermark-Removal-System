FROM tensorflow/tensorflow:1.15.5-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && \
    apt-get -qqq install --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt

RUN git clone https://github.com/JiahuiYu/neuralgym.git /tmp/neuralgym && \
    pip install --no-cache-dir /tmp/neuralgym && \
    rm -rf /tmp/neuralgym && \
    pip install --no-cache-dir opencv-python==4.9.0.80 Pillow numpy pyyaml

RUN adduser nonroot
USER nonroot
COPY --chown=nonroot:nonroot ./ /repo
WORKDIR /repo

ENTRYPOINT ["/usr/bin/env", "python", "/repo/main.py"]
