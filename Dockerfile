# syntax=docker/dockerfile:1
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    bash make curl ca-certificates git sudo \
    build-essential g++ gcc cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

ARG USER=appuser
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USER} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER}

WORKDIR /work
USER ${USER}

CMD ["bash"]
