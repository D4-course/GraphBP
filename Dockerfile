FROM continuumio/miniconda3
ADD GraphBP/environment_cpu.yml /tmp/environment_cpu.yml
RUN mkdir /d4/
RUN conda env create -f /tmp/environment_cpu.yml
RUN pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
RUN pip uninstall -y torch-geometric
RUN pip install torch-geometric==1.7.2
RUN echo "source activate $(head -1 /tmp/environment_cpu.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment_cpu.yml | cut -d' ' -f2)/bin:$PATH
COPY . /d4/
# to make a docker image run in your terminal: docker build -t <name of that image> .
# to run the docker image: docker run -it <name of that image>
# this should open up a prompt with `synthenv` already activated
