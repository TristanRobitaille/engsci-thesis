#!/bin/bash

IMGNAME=engsci-thesis

docker buildx build -f Dockerfile -t $IMGNAME .
docker save $IMGNAME -o $IMGNAME.tar
apptainer build --fakeroot --ignore-subuid $IMGNAME.sif docker-archive://$IMGNAME.tar

rm $IMGNAME.tar
docker image rm $IMGNAME
