# Deep Gradient Compression

We implement the Deep Gradient Compression within the optimizer. We applied the latest feature of the PyTorch 2.0 into the codebase. You can just replace the `SGD` into our `DGC_SGD` to apply the deep gradient compression.

We also introduce the momentum correction into the design so that the optimizer would be avaiable to apply with momentum.

Reference

```tex
@article{lin2017deep,
  title={Deep gradient compression: Reducing the communication bandwidth for distributed training},
  author={Lin, Yujun and Han, Song and Mao, Huizi and Wang, Yu and Dally, William J},
  journal={arXiv preprint arXiv:1712.01887},
  year={2017}
}
```

