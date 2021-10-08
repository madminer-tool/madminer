# References

## Citations

If you use MadMiner, please cite our main publication,
```
@article{Brehmer:2019xox,
      author         = "Brehmer, Johann and Kling, Felix and Espejo, Irina and Cranmer, Kyle",
      title          = "{MadMiner: Machine learning-based inference for particle physics}",
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
      url            = {https://github.com/madminer-tool/madminer}
}
```


The main references for the implemented inference techniques are the following:

- CARL: [1506.02169][ref-arxiv-carl].
- MAF: [1705.07057][ref-arxiv-maf].
- CASCAL, RASCAL, ROLR, SALLY, SALLINO, SCANDAL:
  - [1805.00013][ref-arxiv-madminer-1].
  - [1805.00020][ref-arxiv-madminer-2].
  - [1805.12244][ref-arxiv-madminer-3].
- ALICE, ALICES: [1808.00973][ref-arxiv-alice].


## Acknowledgements

We are immensely grateful to all [contributors][repo-madminer-contrib] and bug reporters! In particular, we would like
to thank Zubair Bhatti, Philipp Englert, Lukas Heinrich, Alexander Held, Samuel Homiller and Duccio Pappadopulo.

The SCANDAL inference method is based on [Masked Autoregressive Flows][ref-arxiv-scandal], where our implementation is
a PyTorch port of the original code by George Papamakarios, available at [this repository][repo-maf-main-page].


[ref-arxiv-alice]: https://arxiv.org/abs/1808.00973
[ref-arxiv-carl]: https://arxiv.org/abs/1506.02169
[ref-arxiv-maf]: https://arxiv.org/abs/1705.07057
[ref-arxiv-madminer-1]: https://arxiv.org/abs/1805.00013
[ref-arxiv-madminer-2]: https://arxiv.org/abs/1805.00020
[ref-arxiv-madminer-3]: https://arxiv.org/abs/1805.12244
[ref-arxiv-scandal]: https://arxiv.org/abs/1705.07057
[repo-madminer-contrib]: https://github.com/madminer-tool/madminer/graphs/contributors
[repo-maf-main-page]: https://github.com/gpapamak/maf
