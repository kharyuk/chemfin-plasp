FROM continuumio/miniconda
MAINTAINER Pavel Kharyuk

ENTRYPOINT [ "/bin/bash", "-c" ]

RUN groupadd -r researcher && useradd -r -g researcher researcher
RUN apt install make

COPY environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN conda env update --file environment.yml \
    && rm -rf /opt/conda/pkgs/*
    
WORKDIR /chemfin-plasp
COPY ./ /chemfin-plasp
RUN chown -R researcher:researcher /chemfin-plasp

USER researcher
ENV PATH /opt/conda/envs/cfps/bin:$PATH

#RUN jupyter notebook --port=5555
