#Application start (Underline)

Create dockerfile with following structure:

```
   FROM ubuntu:latest
   
   # Install
   RUN \
     apt-get update && \
     apt-get install -y openssh-server && \
     apt-get install -y openmpi-bin libopenmpi-dev && \
     apt install -y default-jdk && \
     apt install -y maven
   
   # Add files
   ADD ./miniproject_3 project/
   
   # Set environment variables.
   ENV HOME /project
   
   # Define working directory.
   WORKDIR /project
   
   # Define default command.
   CMD ["bash"]
```

Run next commands:

```
    docker build ./
    
    docker run -it --rm
    
    export OMPI_MCA_btl_vader_single_copy_mechanism=none
    
    mvn -P MPITests-4 test-compile
```
