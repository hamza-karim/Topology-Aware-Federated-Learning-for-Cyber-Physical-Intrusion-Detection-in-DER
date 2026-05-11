FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV PYTHONUNBUFFERED=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# TensorFlow GPU Configuration
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV TF_USE_LEGACY_KERAS=1

WORKDIR /app

COPY . /app/src

WORKDIR /app/src

RUN mkdir -p /app/src/data /app/src/models && \
    find /app/src -maxdepth 1 -name "*.csv" -exec mv {} /app/src/data/ \;

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "/app/src/Client.py"]
