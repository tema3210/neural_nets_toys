# Use the Microsoft dev container as the base image
FROM mcr.microsoft.com/vscode/devcontainers/base:bookworm

# RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# Update package list and install packages required for Android app development

RUN apt-get update -yqq && \
    apt-get install -y qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils && \
    apt-get install -y \
        libgtk-3-dev \
        clang \
        libstdc++6 \
        lib32z1 \
        ninja-build \
        pkg-config \
        curl \
        # watchman \
        # libnotify-bin \
        python3 \
        python3-pip \
        cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the system language to US English
ENV LANG=en_US.UTF-8