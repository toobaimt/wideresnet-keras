# Wide Residual Networks (WideResNets) in Keras
tf.keras adaptation of [wideresnet-50-2](https://arxiv.org/pdf/1605.07146v1.pdf) for ImageNet.

Tested on validation set of ImageNet, achieves 96.34% accuracy (vs 98% of the torch model, probably due to slighly different preprocessing)

# Evaluation
Add the path to ImageNet folder in Line 114. Then run ```python WideResNet.py```

# Acknowledgement
Official [PyTorch implementation](https://github.com/xternalz/WideResNet-pytorch)

Original paper:
```bibtex
@inproceedings{zagoruyko2016wide,
  title={Wide Residual Networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  booktitle={British Machine Vision Conference 2016},
  year={2016},
  organization={British Machine Vision Association}
}
```
