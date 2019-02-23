# References

If you use MadMiner, please cite this code as
```
@misc{MadMiner,
      author         = "Brehmer, Johann and Kling, Felix and Espejo, Irina and Cranmer, Kyle",
      title          = "{MadMiner}",
      doi            = "10.5281/zenodo.1489147",
      url            = {https://github.com/johannbrehmer/madminer}
}
```

For the inference methods, there are three main references. Two introduce most of the methods in a particle physics
setting:
```
@article{Brehmer:2018kdj,
      author         = "Brehmer, Johann and Cranmer, Kyle and Louppe, Gilles and
                        Pavez, Juan",
      title          = "{Constraining Effective Field Theories with Machine
                        Learning}",
      journal        = "Phys. Rev. Lett.",
      volume         = "121",
      year           = "2018",
      number         = "11",
      pages          = "111801",
      doi            = "10.1103/PhysRevLett.121.111801",
      eprint         = "1805.00013",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-ph",
}

@article{Brehmer:2018eca,
      author         = "Brehmer, Johann and Cranmer, Kyle and Louppe, Gilles and
                        Pavez, Juan",
      title          = "{A Guide to Constraining Effective Field Theories with
                        Machine Learning}",
      journal        = "Phys. Rev.",
      volume         = "D98",
      year           = "2018",
      number         = "5",
      pages          = "052004",
      doi            = "10.1103/PhysRevD.98.052004",
      eprint         = "1805.00020",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-ph",
}
```

In addition, the inference techniques are discussed in a more general setting, and the SCANDAL family of methods is
added in:
```
@article{Brehmer:2018hga,
      author         = "Brehmer, Johann and Louppe, Gilles and Pavez, Juan and
                        Cranmer, Kyle",
      title          = "{Mining gold from implicit models to improve
                        likelihood-free inference}",
      year           = "2018",
      eprint         = "1805.12244",
      archivePrefix  = "arXiv",
      primaryClass   = "stat.ML",
      SLACcitation   = "%%CITATION = ARXIV:1805.12244;%%"
}
```

Some inference methods are introduced in other papers, including [CARL](https://arxiv.org/abs/1506.02169),
[Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057), and [ALICE(S)](https://arxiv.org/abs/1808.00973).

## Acknowledgements

We are immensely grateful to all contributors and bug reporters! In particular, we would like to thank Zubair Bhatti,
Alexander Held, and Duccio Pappadopulo. A big thanks to Lukas Heinrich for his help with workflows and Docker
containers.

The SCANDAL inference method is based on [Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057), and our
implementation is a pyTorch port of the original code by George Papamakarios et al., which is available at
[https://github.com/gpapamak/maf](https://github.com/gpapamak/maf).

The [setup.py](https://github.com/johannbrehmer/madminer/blob/master/setup.py) was adapted from
[https://github.com/kennethreitz/setup.py](https://github.com/kennethreitz/setup.py).
