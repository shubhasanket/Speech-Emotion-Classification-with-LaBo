# Speech Emotion Classification with LaBo

Implementation of [Language in a Bottle (LaBo)](https://arxiv.org/abs/2211.11158) for speech emotion recognition, using [CLAP](https://github.com/microsoft/CLAP) instead of CLIP. Evaluated on the [IEMOCAP](https://huggingface.co/datasets/mteb/iemocap) dataset.

## Models

- `clap_probe`: frozen CLAP audio encoder + small MLP classifier
- `e2e`: end-to-end CNN trained on log-mel spectrograms
- `labo`: language-guided concept bottleneck using CLAP (the paper's method)
- `zero_shot`: CLAP zero-shot via text emotion prompts, no training
- `knn`: kNN over frozen CLAP embeddings, no training

## Dependencies

```
pip install -r requirements.txt
```

## Training

```
python train.py \
  --model clap_probe \
  --epochs 15 \
  --lr 3e-4 \
  --batch-size 32 \
  --save-dir checkpoints/clap_probe
```

You can pass `--model e2e` or `--model labo` to train the other models.

## Evaluation (zero-shot and k-NN)

```
python evaluate.py \
  --mode zero_shot \
  --batch-size 16
```

```
python evaluate.py \
  --mode knn \
  --knn-k 5 \
  --batch-size 16
```

## Arguments

**train.py**

- `--model`: `clap_probe`, `e2e`, `labo`
- `--epochs`: number of training epochs
- `--lr`: learning rate
- `--batch-size`: batch size
- `--save-dir`: folder to save checkpoints
- `--device`:  `cuda` or `cpu`

**evaluate.py**

- `--mode`: `zero_shot` or `knn`
- `--knn-k`: number of neighbours for kNN (default: `5`)
- `--batch-size`: batch size
- `--zero-shot-agg`: `single`, `mean`, `max` or `all`

## LaBo concepts

By default `labo` uses the hard-coded concept bank in `models.py`. To use your own LLM-generated concepts, edit the `concepts` dict in `models.py`. The expected format is:

```python
DEFAULT_CONCEPTS = {
    "anger":   ["voice sounds harsh and aggressive", ...],
    "sadness": ["slow heavy speech low energy", ...],
    ...
}
```

