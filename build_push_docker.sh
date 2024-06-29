#!/bin/bash

docker buildx build . --tag fifth-cont --platform linux/amd64    
docker tag fifth-cont europe-west4-docker.pkg.dev/ingka-feed-student-dev/adam-thesis-container/thesis-model
docker push europe-west4-docker.pkg.dev/ingka-feed-student-dev/adam-thesis-container/thesis-model  