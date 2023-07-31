# Kubernetes 101

## Install Kubernetes

Getting started with kubernetes is straightfoward -  [follow this guide](https://kubernetes.io/docs/tasks/tools/)

- Windows docker desktop provides kubernetes installation
- MacOS users should consider: `brew install kubectl`
- Linux users can use their favorite package manager to install (e.g., apt, snap, pacman)

## Configuration File

Kubernetes will require a configuration file to connect to appropriate server. At this time my chosen computing cluster is [Nautilus](https://docs.nationalresearchplatform.org/)
- Getting access required me to follow the steps outlined [here](https://github.com/MU-HPDI/nautilus/wiki/Getting-Started)

The coniguration file can be found [here](https://portal.nrp-nautilus.io/authConfig)
- Note this requires an organization login
- This needs to be placed at location `~/.kube/config` , so one can run something like `mv ~\Downloads\config ~/.kube/config`


1) Create remote accessible docker image 

    - Build docker image locally and push to docker hub (MacOS)

        - docker build --platform linux/x86_64 --rm -t [tag] [filepath]

            - [tag] is a alias for the created docker image

                - naming can be local in format [tag_name] (e.g., testing) 

                - naming can be dockerhub format [namespace/repository_name/tag_name] (e.g., ctvqfq/bunny_ai:testing) 

            - Note that --platform linux/x86_64 compiles for linux x86_64 devices

        - Check image creation

            - docker images 

        - Login to dockerhub

            - docker login

        - Update image tag (Optional)

             - docker tag [tag_name] [namespace/repository_name/tag_name]

                - If one is trying to prepare a local image for dockerhub

        - Push image to dockerhub

            - docker push ctvqfq/bunny_ai:kube_tutorial

        - Pull the docker container from dockerhub (Optional)

            - docker pull docker.io/ctvqfq/bunny_ai:kube_tutorial

    - Build via GitLab CLI (Not sure how to do yet)
        
        - TODO

2) Manage volume storages

    - Create Persistent Volume Claim (PVC) storage

        - kubectl apply -f [pvc_filename]

    - Check all PVCs
    
        - kubectl get pvc

    - Delete PVC(s)
    
        - Delete single pvc
        
            - kubectl delete pvc [pcv_name]
        
        - Delete all pvcs

            - kubectl delete pvc --all

3) Manage kubernetes pod

    - Create kubernetes pod

        - kubectl apply -f [pod_filename]

    - Check details of pod 

        - kubectl describe pod [pod_name]

        - This is useful for seeing creation details (e.g., when is it useable)

    - Check short details regarding all pods

        - kubectl get pods

    - Create temporary kubernetes pod and run it for interactive testing

        - kubectl run [pod_name] --image [path_container_image] --rm --tty -i -- [command]

            - [pod_name] : name of pod 

            - --image : flag for container image

            - [path_container_image] : path to remote accessible docker image (e.g., docker.io/ctvqfq/bunny_ai:kube_tutorial)

            - --rm : flag to delete pod upon pod completion
    
            - --tty & -i : flags for making interactive terminal session

            - [command] : command to execute inside the pod. 

        - Note that this doesn't appear to have configurable hardware support due to depreciation

    - Create configurable kubernetes pod for interactive testing

        - Create 
