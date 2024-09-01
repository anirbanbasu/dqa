# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pull Python on Debian image
FROM python:3.12.5-slim-bookworm

# Upgrade and install basic packages
RUN apt-get update && apt-get -y install build-essential curl

# Create a non-root user
RUN useradd -m -u 1000 app_user

ENV HOME="/home/app_user"

USER app_user
# Set the working directory in the container
WORKDIR $HOME/app

# Install the Rust toolchain, which is required by some Python packages during their building processes
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/app_user/.cargo/bin:${PATH}"

# Copy only the requirements file to take advantage of layering (see: https://docs.cloud.ploomber.io/en/latest/user-guide/dockerfile.html)
COPY ./requirements.txt ./requirements.txt

# Setup Virtual environment
ENV VIRTUAL_ENV="$HOME/app/venv"
RUN python -m venv $VIRTUAL_ENV
RUN $VIRTUAL_ENV/bin/python -m ensurepip
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir -U pip

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir -U -r requirements.txt

# Uninstall things needed only at build time to reduce the image size by about 300MB
# TODO: Explore better size optimisation methods. Note that alpine base image does not actually help in reducing the size of the image.
USER root
RUN apt-get -y remove build-essential curl && apt-get -y autoremove && rm -rf /var/lib/apt/lists/* && apt-get -y clean
RUN rustup self uninstall -y
USER app_user

# Copy the project files
COPY ./*.md ./LICENSE ./
COPY ./.env.docker /.env
COPY ./src/*.py ./src/
# Copy the required assets
COPY ./assets/logo.svg ./assets/

# Expose the port to conect
EXPOSE 7860
# Run the application
ENTRYPOINT [ "/home/app_user/app/venv/bin/python", "src/webapp.py" ]
