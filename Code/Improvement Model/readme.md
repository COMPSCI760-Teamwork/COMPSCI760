## RNN-T

### Environment

- pytorch >= 0.4
- warp-transducer

### Preparation

We use Kaldi for data preparation.

### Train

```bash
python train.py -config config/aishell.yaml
```

### Eval 

```bash
python eval.py -config config/aishell.yaml
```

### Experiments

| **Model** | **DEV(CER)** | **TEST(CER)** |
| --------- | ------------ | ------------- |
| RNN-T     | 10.13        | 11.82         |

