FROM python:3.13-slim
RUN useradd -m talos
USER talos
WORKDIR /home/talos
CMD ["python3"]
