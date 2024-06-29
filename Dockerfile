# Use pytorch GPU base image
FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest

# Copies to container directory 
COPY recommenders ./Recommender-Models/recommenders
COPY setup.py ./Recommender-Models/
COPY scripts ./Recommender-Models/scripts
COPY docker/requirements.txt ./Recommender-Models/

RUN echo "All files successfully copied." 


# Set environment variables
ENV WANDB_API_KEY=local-c7909b7d0de8a98006d0aa999125094b1463a2b4
ENV WANDB_BASE_URL=https://wandb.mlops.ingka.com


# Create .netrc file with the W&B credentials
RUN echo "machine wandb.mlops.ingka.com login adam.walsh@ingka.ikea.com password local-c7909b7d0de8a98006d0aa999125094b1463a2b4" > /root/.netrc

# Set working directory
WORKDIR /Recommender-Models

# Set the execute permission for the shell script
RUN chmod +x ./scripts/IKEA/training/run_experiment.sh


# Install dependencies
RUN pip install -r requirements.txt
RUN pip install .

RUN echo "All dependencies successfully installed." 

# Set the entrypoint to the shell script - yaml GCP adress will be passed here
ENTRYPOINT ["./scripts/IKEA/training/run_experiment.sh"]  

