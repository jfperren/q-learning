### GCloud Compute Commands

### Create CPU Instance

```
gcloud compute instances create "pytorch-cpu" \
  --zone="us-west1-b" \
  --image-family="pytorch-latest-cpu" \
  --image-project="deeplearning-platform-release" \
  --project="reinforcement-learning-276000"
```

### Connect Instance to JupyterLab

```
gcloud compute ssh \
--project "reinforcement-learning-276000" \
--zone "us-west1-b" \
  pytorch-cpu -- -L 8080:localhost:8080
```

Then, connect to `http://localhost:8080`.

### Install Requires Dependencies

```
pip install tensorflow, gym
```

#### Setting up Tensorboard

```
pip install jupyter_tensorboard
```

```
jupyter labextension install jupyterlab_tensorboard
```

### Add Project & Path

```
git clone https://github.com/jfperren/q-learning
```

Then, in the notebook,

```
import sys
sys.path.append('/home/jupyter/q-learning/')
```
