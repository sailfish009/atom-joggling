# Supercon

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/supercon/main.svg)](https://results.pre-commit.ci/latest/github/janosh/supercon/main)

[CGCNN](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301) + [Materials Project](https://materialsproject.org/) + [MixMatch](https://arxiv.org/abs/1905.02249) for searching superconductors

## Roadmap

- take Yunwei's 50 known BCS superconductors and 50 known non-superconductors plus the roughly 7000 unlabelled structures from materials project
- build a binary classifier from the labelled materials
- and then train the model with self-supervised mixmatch using the unlabelled structures
