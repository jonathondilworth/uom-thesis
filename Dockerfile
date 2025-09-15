# syntax=docker/dockerfile:1.6

ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE} as base

ARG GPU=1
ARG ENV_NAME=knowledge-retrieval-env

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 LANG=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    ENV_NAME=${ENV_NAME}

ENV AUTO_ENV_NAME=knowledge-retrieval-env

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git jq wget unzip build-essential \
    openjdk-17-jdk maven \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p "$CONDA_DIR" \
 && rm -f /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda config --set always_yes yes --set changeps1 no \
 && conda config --set auto_activate_base false

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /tmp/build
COPY environment.yml ./
RUN conda env create -n "${ENV_NAME}" -f environment.yml \
 || conda env update -n "${ENV_NAME}" -f environment.yml --prune

RUN if [ "${GPU}" = "1" ]; then \
      conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1 ; \
    else \
      conda run -n "${ENV_NAME}" python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1 ; \
    fi

COPY requirements.txt ./
RUN if [ -f requirements.txt ]; then \
      conda run -n "${ENV_NAME}" python -m pip install --upgrade pip && \
      conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt ; \
    fi

RUN conda run -n "${ENV_NAME}" python -m pip install \
      phonemizer pysbd sentencepiece geoopt deeponto latextable scispacy --no-deps


ARG ROBOT_VERSION=1.9.7
WORKDIR /opt/robot
RUN curl -fsSL -o robot.jar \
      "https://github.com/ontodev/robot/releases/download/v${ROBOT_VERSION}/robot.jar" \
 && printf '%s\n' '#!/usr/bin/env bash' \
                  'exec java -jar /opt/robot/robot.jar "$@"' \
    > /usr/local/bin/robot \
 && chmod +x /usr/local/bin/robot
ENV ROBOT_JAR=/opt/robot/robot.jar

# sanity check
RUN robot --version

# DOES NOT WORK AS INTENDED WITHIN DOCKER CONTAINER:
#RUN git clone https://github.com/ontodev/robot.git \
# && cd robot \
# && mvn clean package

WORKDIR /work
COPY pyproject.toml* setup.cfg* setup.py* ./
COPY lib/ ./lib/
COPY src/ ./src/
RUN conda run -n "${ENV_NAME}" python -m pip install -e .

ENV TRANSFORMERS_NO_TORCHVISION=1

SHELL ["/bin/bash", "-lc"]
RUN echo "source \"$CONDA_DIR/etc/profile.d/conda.sh\" && conda activate \"${ENV_NAME}\"" >> /etc/profile.d/activate.sh
ENV PATH=$CONDA_DIR/envs/${ENV_NAME}/bin:$PATH
ENV AUTO_ENV_NAME=${ENV_NAME}
RUN echo "AUTO_ENV_NAME=${ENV_NAME}" >> .env

RUN useradd -m -s /bin/bash appuser && chown -R appuser:appuser /work
USER appuser

WORKDIR /work
CMD ["bash"]
