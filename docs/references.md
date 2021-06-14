# References

## Citations

If you use MadMiner, please cite our main publication,
```
@article{Brehmer:2019xox,
      author         = "Brehmer, Johann and Kling, Felix and Espejo, Irina and
                        Cranmer, Kyle",
      title          = "{MadMiner: Machine learning-based inference for particle
                        physics}",
      journal        = "Comput. Softw. Big Sci.",
      volume         = "4",
      year           = "2020",
      number         = "1",
      pages          = "3",
      doi            = "10.1007/s41781-020-0035-2",
      eprint         = "1907.10621",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-ph",
      SLACcitation   = "%%CITATION = ARXIV:1907.10621;%%"
}
```

The code itself can be cited as
```
@misc{MadMiner_code,
      author         = "Brehmer, Johann and Kling, Felix and Espejo, Irina and Cranmer, Kyle",
      title          = "{MadMiner}",
      doi            = "10.5281/zenodo.1489147",
      url            = {https://github.com/diana-hep/madminer}
}
```


The main references for the implemented inference techniques are the following:

- CARL: [1506.02169](https://arxiv.org/abs/1506.02169)
- MAF: [1705.07057](https://arxiv.org/abs/1705.07057)
- CASCAL, RASCAL, ROLR, SALLY, SALLINO, SCANDAL: [1805.00013](https://arxiv.org/abs/1805.00013), [1805.00020](https://arxiv.org/abs/1805.00020), [1805.12244](https://arxiv.org/abs/1805.12244)
- ALICE, ALICES: [1808.00973](https://arxiv.org/abs/1808.00973)


## Acknowledgements

We are immensely grateful to all contributors and bug reporters! In particular, we would like to thank Zubair Bhatti,
Philipp Englert, Lukas Heinrich, Alexander Held, Samuel Homiller, and Duccio Pappadopulo.

The SCANDAL inference method is based on [Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057), and our
implementation is a pyTorch port of the original code by George Papamakarios et al., which is available at
[https://github.com/gpapamak/maf](https://github.com/gpapamak/maf).

The `setup.py` was adapted from [https://github.com/kennethreitz/setup.py](https://github.com/kennethreitz/setup.py).
