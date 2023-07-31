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

