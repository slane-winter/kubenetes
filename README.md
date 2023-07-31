# Kubernetes 101

## Install Kubernetes

Getting started with kubernetes is straightfoward -  [follow this guide](https://kubernetes.io/docs/tasks/tools/)

- Windows docker desktop provides kubernetes installation
- MacOS users should consider: `brew install kubectl`
- Linux users can use their favorite package manager to install (e.g., apt, snap, pacman)

## Configuration File

Kubernetes will require a configuration file to connect to appropriate server. At this time, the chosen computing cluster is [Nautilus](https://dash.nrp-nautilus.io/)
- Getting access required following the steps outlined [here](https://github.com/MU-HPDI/nautilus/wiki/Getting-Started)

The coniguration file can be found [here](https://portal.nrp-nautilus.io/authConfig)
- Note this requires an organization login
- This needs to be placed at location `~/.kube/config` , so one can run something like `mv ~\Downloads\config ~/.kube/config`

To verify configuration is setup properly one can run the following:
- `kubectl get pods` : shows all accessible pods attached to the determined namespace
- `kubectl config view --minify -o jsonpath='{..namespace}'` : shows the users namespace

## Kubernetes Examples

If an example isn't covererd directly here, check out [this link](https://docs.nationalresearchplatform.org/).

First, its important to [read about some basic terminology](https://www.vmware.com/topics/glossary/content/components-kubernetes.html).

### Create Ineractive Kubernetes Environment

This will create a pod with requested resouces (e.g., GPUs, CPUs, RAM). 
- It revolves around 4 steps
    - Create the docker image
    - Define storage details via yaml
    - Define pod details via yaml
    - Enter pod container

Create Docker Image (Remotely Accessible)
- Build docker image locally and push to docker hub: `docker build --platform linux/x86_64 --rm -t [tag] [filepath]`
    - Note that `--platform linux/x86_64` compiles for linux x86_64 devices
    - `[tag]` : docker image alias, specified as local format `[tag_name]` or dockerhub format `[namespace/repository_name:tag_name]`
        - Tag must be updated via `docker tag [tag_name] [namespace/repository_name:tag_name]` if not in dockerhub format. 

    - Check image creation: `docker images`
    - Login to [dockerhub](https://hub.docker.com/): `docker login`
    - Push image to dockerhub: `docker push [namespace/repository_name:tag_name]`

Define Storage Details as Persistent Volume Claim (PVC) storage
- Using `data.yaml` or `result.yaml` as examples, create a custom yaml (a.k.a. `[pvc_filename]`) for data storage.
- Afterwards, initialize the storage via : `kubectl apply -f [pvc_filename]`
- Addtional options are the following
  - Check all PVCs : `kubectl get pvc`
  - Delete single pvc : `kubectl delete pvc [pcv_name]`
  - Delete all pvcs : `kubectl delete pvc --all`

Define Pod Details
- Using `pod.yaml` as an example, createa a custom yaml for a pod. 
- Afterwards, initialize the storage via : `kubectl apply -f [pod_filename]`
    - Pod creation can take some time. Check verbose details of pod via : `kubectl describe pod [pod_name]`
    - Check short details regarding all pods : `kubectl get pods`

Enter Pod Container
- `kubectl exec -it [pod_name] -- /bin/bash`
