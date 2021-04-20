# Atom Joggling

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/atom-joggling/main.svg)](https://results.pre-commit.ci/latest/github/janosh/atom-joggling/main)

## Results

Tried Atom Joggling as a method for data augmentation with [Rhys'es CGCNN implementation](https://github.com/CompRhys/roost/tree/master/roost/cgcnn) on 6 out of 9 MatBench datasets with structures:

- `matbench_dielectric`
- `matbench_jdft2d`
- `matbench_log_gvrh`
- `matbench_log_kvrh`
- `matbench_perovskites`
- `matbench_phonons`

Worked best when splitting train and test set by spacegroups. Showed no performance difference at all on random splits.

With spacegroup splits, the test set performance improvement was most pronounced on the `matbench_phonons` dataset, very subtle on the `matbench_log_g/kvrh`, `matbench_jdft2d` and `matbench_perovskites` datasets and non-existent on `matbench_dielectric`. See [`tensorboard_screenshots`](tensorboard_screenshots) and the following uploads of TensorBoard logs for results.

## TensorBoard.dev Experiments

- [fine-tuning the XXL CGCNN for 200 epochs with 3e-5 learning rate after initial training of 600 epochs with lr 3e-4 on MatBench Perovskites](https://tensorboard.dev/experiment/K9e8Zp78QnC4ZIZIgeZsxg): Joggling amplitudes = [0, 0.01, 0.02, 0.03] A, joggling rate = 0.3, test set = spacegroups 200-231
- [Different-sized CGCNN models tested on MatBench Perovskites](https://tensorboard.dev/experiment/a9SLIwliRKOjlqUD9ho7ow): Joggling amplitudes = [0, 0.01, 0.02, 0.03] A, joggling rate = 0.3, test set = spacegroups 200-231
  - [Same experiment but only uploaded the mean of 10 repeats for each parameter set of the different CGCNN models](https://tensorboard.dev/experiment/a9SLIwliRKOjlqUD9ho7ow)
- [MatBench Phonons dataset (last phonon DOS peak)](https://tensorboard.dev/experiment/eK1K6de2Q9OO4ielI2mBjQ/#scalars&regexInput=-mean%24)
- [Shear Modulus showing less of a benefit with joggling than MatBench Phonons does](https://tensorboard.dev/experiment/pyAacvKjQTqDebqvVWiEeQ)
  - [Same dataset without 10 repeats for each joggling rate](https://tensorboard.dev/experiment/X0eA5iXdQlqfJ0gS7Tk5ZA)
- [MatBench JDFT (exfoliation energy)](https://tensorboard.dev/experiment/Brk3m7LqSde1QN7ofkVpDg)

## List of MatBench datasets

| task name                | target column (unit)         | sample count | task type      | input       | links                             |
| ------------------------ | ---------------------------- | ------------ | -------------- | ----------- | --------------------------------- |
| `matbench_dielectric`    | `n` (unitless)               | 4764         | regression     | structure   | [download][1], [interactive][2]   |
| `matbench_expt_gap`      | `gap expt` (eV)              | 4604         | regression     | composition | [download][3], [interactive][4]   |
| `matbench_expt_is_metal` | `is_metal` (unitless)        | 4921         | classification | composition | [download][5], [interactive][6]   |
| `matbench_glass`         | `gfa` (unitless)             | 5680         | classification | composition | [download][7], [interactive][8]   |
| `matbench_jdft2d`        | `exfoliation_en` (meV/atom)  | 636          | regression     | structure   | [download][9], [interactive][10]  |
| `matbench_log_gvrh`      | `log10(G_VRH)` (log(GPa))    | 10987        | regression     | structure   | [download][11], [interactive][12] |
| `matbench_log_kvrh`      | `log10(K_VRH)` (log(GPa))    | 10987        | regression     | structure   | [download][13], [interactive][14] |
| `matbench_mp_e_form`     | `e_form` (eV/atom)           | 132752       | regression     | structure   | [download][15], [interactive][16] |
| `matbench_mp_gap`        | `gap pbe` (eV)               | 106113       | regression     | structure   | [download][17], [interactive][18] |
| `matbench_mp_is_metal`   | `is_metal` (unitless)        | 106113       | classification | structure   | [download][19], [interactive][20] |
| `matbench_perovskites`   | `e_form` (eV, per unit cell) | 18928        | regression     | structure   | [download][21], [interactive][22] |
| `matbench_phonons`       | `last phdos peak` (1/cm)     | 1265         | regression     | structure   | [download][23], [interactive][24] |
| `matbench_steels`        | `yield strength` (MPa)       | 312          | regression     | composition | [download][25], [interactive][26] |

[1]: https://ml.materialsproject.org/projects/matbench_dielectric.json.gz
[2]: https://ml.materialsproject.org/projects/matbench_dielectric
[3]: https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz
[4]: https://ml.materialsproject.org/projects/matbench_expt_gap
[5]: https://ml.materialsproject.org/projects/matbench_expt_is_metal.json.gz
[6]: https://ml.materialsproject.org/projects/matbench_expt_is_metal
[7]: https://ml.materialsproject.org/projects/matbench_glass.json.gz
[8]: https://ml.materialsproject.org/projects/matbench_glass
[9]: https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz
[10]: https://ml.materialsproject.org/projects/matbench_jdft2d
[11]: https://ml.materialsproject.org/projects/matbench_log_gvrh.json.gz
[12]: https://ml.materialsproject.org/projects/matbench_log_gvrh
[13]: https://ml.materialsproject.org/projects/matbench_log_kvrh.json.gz
[14]: https://ml.materialsproject.org/projects/matbench_log_kvrh
[15]: https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz
[16]: https://ml.materialsproject.org/projects/matbench_mp_e_form
[17]: https://ml.materialsproject.org/projects/matbench_mp_gap.json.gz
[18]: https://ml.materialsproject.org/projects/matbench_mp_gap
[19]: https://ml.materialsproject.org/projects/matbench_mp_is_metal.json.gz
[20]: https://ml.materialsproject.org/projects/matbench_mp_is_metal
[21]: https://ml.materialsproject.org/projects/matbench_perovskites.json.gz
[22]: https://ml.materialsproject.org/projects/matbench_perovskites
[23]: https://ml.materialsproject.org/projects/matbench_phonons.json.gz
[24]: https://ml.materialsproject.org/projects/matbench_phonons
[25]: https://ml.materialsproject.org/projects/matbench_steels.json.gz
[26]: https://ml.materialsproject.org/projects/matbench_steels

## MatBench Leaderboard

| task name                | verified top score (MAE or ROCAUC) | algorithm name, config,             | general purpose algorithm? |
| ------------------------ | ---------------------------------- | ----------------------------------- | -------------------------- |
| `matbench_dielectric`    | 0.299 (unitless)                   | Automatminer express v1.0.3.2019111 | yes                        |
| `matbench_expt_gap`      | 0.416 eV                           | Automatminer express v1.0.3.2019111 | yes                        |
| `matbench_expt_is_metal` | 0.92                               | Automatminer express v1.0.3.2019111 | yes                        |
| `matbench_glass`         | 0.861                              | Automatminer express v1.0.3.2019111 | yes                        |
| `matbench_jdft2d`        | 38.6 meV/atom                      | Automatminer express v1.0.3.2019111 | yes                        |
| `matbench_log_gvrh`      | 0.0849 log(GPa)                    | Automatminer express v1.0.3.2019111 | yes                        |
| `matbench_log_kvrh`      | 0.0679 log(GPa)                    | Automatminer express v1.0.3.2019111 | yes                        |
| `matbench_mp_e_form`     | 0.0327 eV/atom                     | MEGNet v0.2.2                       | yes, structure only        |
| `matbench_mp_gap`        | 0.228 eV                           | CGCNN (2019)                        | yes, structure only        |
| `matbench_mp_is_metal`   | 0.977                              | MEGNet v0.2.2                       | yes, structure only        |
| `matbench_perovskites`   | 0.0417                             | MEGNet v0.2.2                       | yes, structure only        |
| `matbench_phonons`       | 36.9 cm^-1                         | MEGNet v0.2.2                       | yes, structure only        |
| `matbench_steels`        | 95.2 MPa                           | Automatminer express v1.0.3.2019111 | yes                        |
