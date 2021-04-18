# Toxicity Detection

This source code provides an implementation of content moderation, especially on toxicity detection. The goal is to help highlight toxic content while considering the main context of the text.

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

## Pipelines

### 1. Toxicity Prediction

**Data:** Jigsaw Unintended Bias in Text Classification. Train Size: 800k; Test Size: 100k;

**Text Preprocessing:** Clean special characters and add intonation mark

**Model:** BERT-base-uncased

**Hyperparameters:**
- Learning rate: 2 x 10-5  
- Batch size: 16 </br>(in some kaggle discussion, the higher batch size is recommended. But due to GPU memory limitation, the maximal batch size that I can use only 16)
- Max sequence length: 320
- Num epoch: 1

**Output:** toxicity score, i.e. a real number in between 0 - 1. 1 means the highest toxic. 0 means non-toxic.

**Code:** `working/train-toxicity.ipynb` and `working/predict-toxicity.ipynb`

### 2. Sentiment Analysis

**Data:**  SemEval 2017 (Sentiment Analysis in Twitter)

**Text Preprocessing:** Remove symbol (@ and #) and remove url

**Model:** RoBERTa-base

**Output:** sentiment scores for positive, negative, and neutral labels

**Note:** I use a fine-tuned model available at HuggingFace provided by (Barbieri, 2020). The model is available at https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment 

**Code:** `working/sentiment-analysis.ipynb` 


### 3. Combine Toxicity Prediction and Sentiment Analysis

Since this part is still a grey area, currently I propose a heuristic approach based on some predefined thresholds. These thresholds can be adjusted based on the business usage. Let's say we want to filter more text, then we increase the threshold to make the model more strict to the user input. In case our model is too strict, we can loosen the thresholds. This threshold must be adjustable as filtering unintended texts may make the user frustrated and move to another app. Also make very loose thresholds make the model miss many toxic tweets/comments.

**Code:** `working/combine-prediction.ipynb` 

## Simple Deployment

I built a RESTful API using flask. To deploy the API, please run `bash run_flask.sh` from `working` folder. This API will receive a sentence and output the combined score from toxic prediction. For the demo purpose, I add some outputs about the analysis on combining the toxic prediction model and the sentiment analysis model. 

```json
{
    "analysis": {
        "toxicity": "0.6695",
        "sentiment": "{'negative': 0.0973, 'neutral': 0.1752, 'positive': 0.7275}",
        "edge_case": [
            {
                "truncated_sentence": "What an awesome goal. I nearly missed it…",
                "toxicity": "0.0035",
                "sentiment": "{'negative': 0.0176, 'neutral': 0.0638, 'positive': 0.9186}"
            }
        ]
    },
    "combined_score": "0.0035"
}
```

The JSON above shows the API response when given a query **“Oh sh*t!! What an awesome goal, I nearly missed it…”**. If we use only the toxicity prediction, the toxicity score is quite high (0.6695) which is not correct. The combined score is 0.0035, it indicates that our model can handle this case.

