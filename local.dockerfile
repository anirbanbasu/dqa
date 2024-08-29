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
FROM python:3.13.0rc1-slim-bookworm

# Upgrade and install basic packages
RUN apt-get update && apt-get -y upgrade && apt-get -y install build-essential && apt-get -y autoremove

# Create a non-root user
RUN useradd -m -u 1000 app_user

ENV HOME="/home/app_user"

USER app_user
# Set the working directory in the container
WORKDIR $HOME/app

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
