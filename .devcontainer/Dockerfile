# syntax=docker/dockerfile:1
#
# rune GPU devcontainer — bakes node, uv, claude-code, gh, aws-cli, python 3.12
# into one cacheable layer instead of running 5 devcontainer features per
# `devpod up`. Each feature install was ~30-60s of unpack-and-configure;
# combined they added ~2-3 min and ~10000 lines of log output to every
# devcontainer creation. With this image cached by the host's containerd
# snapshotter, repeated `devpod up` calls skip the install entirely.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Single big RUN: smaller image, fewer cache breaks vs. one-per-tool.
# - python 3.12 from deadsnakes PPA (rune requires-python = ">=3.12,<3.13")
# - gh CLI from official apt repo
# - AWS CLI v2 from upstream zip; -q on unzip silences the ~10k example .rst
#   files that the devcontainer feature was logging on every create.
# - Node.js LTS from nodesource (for claude-code).
# - claude-code via npm.
# - uv from Astral installer; copy to /usr/local/bin so it's on PATH for
#   non-login shells (the devcontainer feature install couldn't do this).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg software-properties-common \
      git openssh-client unzip sudo zsh && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && apt-get install -y --no-install-recommends gh && \
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip && \
    unzip -q /tmp/awscli.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/aws /tmp/awscli.zip && \
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g @anthropic-ai/claude-code && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    cp /root/.local/bin/uv /usr/local/bin/uv && \
    cp /root/.local/bin/uvx /usr/local/bin/uvx && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Non-root vscode user — mirrors the `common-utils:2` feature behaviour so
# the existing devcontainer.json `remoteUser: vscode` keeps working.
RUN groupadd --gid 1000 vscode && \
    useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode && \
    chmod 0440 /etc/sudoers.d/vscode
