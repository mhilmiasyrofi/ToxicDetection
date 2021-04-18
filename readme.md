# Toxic Detection


## Preparation

### Environment

Pull docker image as our development environment. This image is a [Kaggle Python GPU](https://github.com/Kaggle/docker-python) docker image. This image contains all libraries needed in to develop the solution

```bash
docker pull gcr.io/kaggle-gpu-images/python@sha256:36cf6c012f2c2a866c63ef04c13567cd34b2097c27ae22eb096a8f2f0da7d82b
```

Run docker as a container
```bash
docker run --name toxic-prediction --rm --gpus '"device=0"' -it -p 8080:8080  -v  <path to ToxicDetection>:/home/jupyter/ToxicDetection/ gcr.io/kaggle-gpu-images/python@sha256:36cf6c012f2c2a866c63ef04c13567cd34b2097c27ae22eb096a8f2f0da7d82b
```

### Dataset and Final Model
#### Download Jigsaw Toxicity Detection Dataset

You can download manually the dataset from [https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data). Alternatively, you can follow my way to download the dataset.
1. Setup Kaggle Environment. Download `kaggle.json` config from your Kaggle account. Run `bash setup.sh` insed `kaggle/kaggle.json` folder
2. Download dataset by run `bash download.sh` inside `input/jigsaw-unintended-bias-in-toxicity-classification/` folder

#### Final Model
Please follow the instruction in `input/fine-tuned-model/` folder to download the final model


