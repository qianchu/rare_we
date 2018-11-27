FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt install build-essential && \
    apt-get install -y wget bzip2 ca-certificates curl git vim bzip2 bash sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    chmod -R 777 /opt/conda/

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ADD start.sh /home/start.sh
ADD environment_rare_we_clean.yml /home/environment_rare_we_clean.yml
RUN chmod +x /home/start.sh

RUN useradd -m ql261 && echo "ql261:ql261" | chpasswd && adduser ql261 sudo && echo "root:ql261" | chpasswd

ENTRYPOINT [ "/usr/bin/tini","--" ]
CMD [ "/bin/bash" ]