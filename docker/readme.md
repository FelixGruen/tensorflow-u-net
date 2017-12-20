The Docker image built from the files in this directory, creates liver lesion prediction volumes for all medical volumes in a given directory. When started, the Docker container loads all NiBabel volumes in the given directory one-by-one, segments, first, the liver and, second, the liver lesions and saves the liver lesion segmentations in NiBabel volumes in the same directory before closing itself.

This particular directory contains all the files needed to build the docker image, except the saved models themselves which have to be downloaded separately.

## Building the Docker Image

Download the saved model files from [OneDrive](https://1drv.ms/f/s!AgZ-sMRclB9bokSYBQFdDkAethTw) and place the folder "models" containing the saved models inside the folder "build_files".

Then switch to the "docker" directory, where the Dockerfile is placed, and run

    docker build -t username/imagename .

Use the -t flag to give your newly created Docker image a name.

## Use the Docker Image

Use Docker run with the path to the directory that contains the medical volumes as the last argument. Don't forget to mount the directory in the container using the -v flag.

    nvidia-docker run -ti -v /path/to/:/path/to/ username/imagename /path/to/my/data/
