## Model-Agnostic Meta-Learning Pytorch Implementation

Original implementation by Tristan Deleu.

Usage:
Train:

```
python train.py --config configs/maml/halfcheetah-vel.yaml --output-folder maml-halfcheetah-vel --seed 1 --num-workers 8 --use-cuda
```
Test:

```
python test.py --config maml-halfcheetah-vel/config.json --policy maml-halfcheetah-vel/policy.th --output maml-halfcheetah-vel/results.npz --num-batches 10 --meta-batch-size 20 --num-workers 12 --use-cuda
```

Draw:

```
python draw.py --resultpath maml-halfcheetah-vel --num-batches 10 --num-traj 20 
```