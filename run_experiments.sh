python train.py \
  --model clap_probe \
  --epochs 15 \
  --lr 3e-4 \
  --batch-size 32 \
  --save-dir checkpoints/clap_probe

python train.py \
  --model e2e \
  --epochs 15 \
  --lr 3e-4 \
  --batch-size 32 \
  --save-dir checkpoints/e2e

python train.py \
  --model labo \
  --epochs 15 \
  --lr 3e-4 \
  --batch-size 32 \
  --save-dir checkpoints/labo_balance

python evaluate.py \
  --mode zero_shot \
  --zero-shot-agg "all" \
  --batch-size 16

python evaluate.py \
  --mode knn \
  --knn-k 5 \