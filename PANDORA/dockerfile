FROM continuumio/miniconda3

WORKDIR /app

#test
RUN conda update -n base -c defaults conda 

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "conda activate pandora" >> ~/.bashrc

SHELL ["conda", "run", "-n", "pandora", "/bin/bash", "-c"]

# Please replace "XXXX" with your MODELLER license key
ENV KEY_MODELLER XXXX

# Install the package
RUN conda install -c csb-nijmegen csb-pandora -c salilab -c bioconda
RUN python -c "import PANDORA"
RUN echo -e "from PANDORA import Database \nDatabase.create_db_folders('~/PANDORA_databases/default')" > install_db.py
RUN python install_db.py

### Install tcsh to make netMHCpan running
# RUN apt-get update
# RUN apt-get install tcsh

### Copy and install netMHCpan
# COPY netMHCpan-4.1/ ./netMHCpan-4.1/
# ENV PATH="${PATH}:/app/netMHCpan-4.1"

### Copy and install netMHCIIpan
# COPY netMHCIIpan-4.1/ ./netMHCIIpan-4.1/
# ENV PATH="${PATH}:/app/netMHCIIpan-4.1"

