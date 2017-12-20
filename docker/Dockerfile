FROM gcr.io/tensorflow/tensorflow:1.3.0-gpu
ADD build_files /compute_lesion_predictions/
RUN pip install nibabel scipy
ENTRYPOINT ["python", "/compute_lesion_predictions/generate_volumes.py"]
