### GCloud Compute Commands

### Create CPU Instance

```
gcloud compute instances create "pytorch-cpu" \
  --zone="us-west1-b" \
  --image-family="pytorch-latest-cpu" \
  --image-project="deeplearning-platform-release" \
  --project="reinforcement-learning-276000"
```
