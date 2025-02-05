FROM doduo1.umcn.nl/uokbaseimage/diag:tf2.8-pt1.10-v1

ENV CONDA_ENV_NAME=multiscale
ENV CODE_DIRECTORY=/home/user/code/multiscale
ENV PYTHON_VERSION=3.8

# Fix: public key error for apt update
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub


# Basic setup
RUN apt update
RUN apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists


# Copy source
COPY . ${CODE_DIRECTORY}

# Set working directory
WORKDIR ${CODE_DIRECTORY}

RUN chown -R user:user ${CODE_DIRECTORY}

# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /conda \
    && echo export PATH=/conda/bin:$PATH >> .bashrc \
    && rm miniconda3.sh
ENV PATH="/conda/bin:${PATH}"
RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}


# Switch to bash shell
SHELL ["/bin/bash", "-c"]


# Install requirements
COPY requirements.txt ./
RUN source activate ${CONDA_ENV_NAME} \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt \
    && pip install -e .


COPY docker/.bash_aliases /root/


#### Configure entrypoint.
COPY docker/run.sh /root/
ENTRYPOINT ["/bin/bash", "/root/run.sh"]
