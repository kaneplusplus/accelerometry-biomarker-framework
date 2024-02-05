# docker buildx build --platform linux/amd64 -t kaneplusplus/r-torch-x86:latest --push .
# docker buildx build --progress=plain --platform linux/amd64 -t r-torch-x86 . 

# Nvidia drivers
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

FROM rocker/tidyverse:latest
LABEL maintainter="michael.kane@yale.edu"

#SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get upgrade -y

RUN export LC_ALL=en_US.UTF-8
RUN export LANG=en_US.UTF-8

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install locales locales-all apt-utils linux-headers-$(uname -r) open-mpi
RUN locale-gen en_US.UTF-8
RUN dpkg-reconfigure locales

RUN apt-get -y install gpg smbclient smbmap
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get -y install keyboard-configuration
RUN apt-get install -y software-properties-common


RUN export OS=2004
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin 
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm -rf /etc/apt/sources.list.d/cuda.list 
RUN apt-get update -y

#RUN mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
#RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/7fa2af80.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
#RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

RUN apt-get update -y

#RUN export cudann_version=8.6.0.163
#RUN export cuda_version=11.3.0

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
RUN dpkg -i libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
RUN dpkg -i libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb

RUN Rscript -e "remotes::install_cran('torch')"
RUN Rscript -e "remotes::install_cran('arrow')"
RUN Rscript -e "torch::install_torch(type = '11.3')"
RUN Rscript -e "remotes::install_cran('Rmpi')"
RUN Rscript -e "remotes::install_cran('doMPI')"
RUN Rscript -e "remotes::install_cran('haven')"
RUN Rscript -e "remotes::install_cran('tidyverse')"
RUN Rscript -e "remotes::install_cran('ggplot2')"

