# Collection of `sbatch` Commands used to Joggling Experiments

```sh
for joggle_std in 0 0.01 0.02 0.03
do
  sbatch -J joggle=$joggle_std-sg_split --array 0-9 -t 6:0:0 --export CMD="python examples/cgcnn-sg-split.py --train --evaluate --log --data-name matbench_perovskites --targets 'e_form' --tasks regression --losses L1 --epochs 600 --model-name perovskites-sg_split=200-231-joggle_std=$joggle_std-joggle_rate=0.3 --joggle-std $joggle_std --joggle-rate 0.3 --sg-split 200-231 --run-id \$SLURM_ARRAY_TASK_ID" hpc/gpu_submit
done
```

```sh
for joggle_std in 0 0.01 0.02 0.03
do
  sbatch -J joggle=$joggle_std-sg_split --array 0-9 -t 1:0:0 --export CMD="python examples/cgcnn-sg-split.py --train --evaluate --log --data-name matbench_jdft2d --targets 'exfoliation_en' --tasks regression --losses L1 --epochs 600 --model-name jdft2d-sg_split=75-143-joggle_std=$joggle_std-joggle_rate=0.3 --joggle-std $joggle_std --joggle-rate 0.3 --sg-split 75-143 --run-id \$SLURM_ARRAY_TASK_ID" hpc/gpu_submit
done
```

## XXL CGCNN

- 4,406,785 params

```sh
for joggle_std in 0 0.01 0.02 0.03
do
  sbatch -J joggle=$joggle_std-sg_split --array 0-9 -t 12:0:0 --export CMD="python examples/cgcnn-sg-split.py --train --evaluate --log --data-name matbench_perovskites --targets 'e_form' --tasks regression --losses L1 --epochs 600 --model-name perovskites-xxl_cgcnn-sg_split=200-231-joggle_std=$joggle_std-joggle_rate=0.3 --joggle-std $joggle_std --joggle-rate 0.3 --sg-split 200-231 --run-id \$SLURM_ARRAY_TASK_ID --elem-fea-len 256 --h-fea-len 256 --n-graph 12 --n-hidden 8" hpc/gpu_submit
done
```

## Large CGCNN

- 681,857 params

```sh
for joggle_std in 0 0.01 0.02 0.03
do
  sbatch -J joggle=$joggle_std-sg_split --array 0-9 -t 6:0:0 --export CMD="python examples/cgcnn-sg-split.py --train --evaluate --log --data-name matbench_perovskites --targets 'e_form' --tasks regression --losses L1 --epochs 600 --model-name perovskites-big_cgcnn-sg_split=200-231-joggle_std=$joggle_std-joggle_rate=0.3 --joggle-std $joggle_std --joggle-rate 0.3 --sg-split 200-231 --run-id \$SLURM_ARRAY_TASK_ID --elem-fea-len 128 --h-fea-len 256 --n-graph 6 --n-hidden 2" hpc/gpu_submit
done
```

## Small CGCNN

### Perovskites

```sh
for joggle_std in 0 0.01 0.02 0.03
do
  sbatch -J joggle=$joggle_std-sg_split --array 0-9 -t 6:0:0 --export CMD="python examples/cgcnn-sg-split.py --train --evaluate --log --data-name matbench_perovskites --targets 'e_form' --tasks regression --losses L1 --epochs 600 --model-name perovskites-small_cgcnn-sg_split=200-231-joggle_std=$joggle_std-joggle_rate=0.3 --joggle-std $joggle_std --joggle-rate 0.3 --sg-split 200-231 --run-id \$SLURM_ARRAY_TASK_ID --elem-fea-len 32 --h-fea-len 64 --n-graph 2 --n-hidden 1" hpc/gpu_submit
done
```

### Phonons

```sh
for joggle_std in 0 0.01 0.02 0.03
do
  sbatch -J joggle=$joggle_std-sg_split --array 0-9 -t 2:0:0 --export CMD="python examples/cgcnn-sg-split.py --train --evaluate --log --data-name matbench_phonons --targets 'last phdos peak' --tasks regression --losses L1 --epochs 600 --model-name phonons-small_cgcnn-sg_split=1-100-joggle_std=$joggle_std-joggle_rate=0.3 --joggle-std $joggle_std --joggle-rate 0.3 --sg-split 1-100 --run-id \$SLURM_ARRAY_TASK_ID --elem-fea-len 32 --h-fea-len 64 --n-graph 2 --n-hidden 1" hpc/gpu_submit
done
```

## Single run for testing

```sh
python examples/cgcnn-sg-split.py --train --evaluate --log --data-name matbench_perovskites --targets 'e_form' --tasks regression --losses L1 --epochs 600 --model-name perovskites-sg_split-joggle_std=0.01-joggle_rate=0.3 --joggle-std 0.01 --joggle-rate 0.3 --sg-split 200-231
```

```sh
for joggle_std in 0 0.01 0.02 0.03
do
  tb-reducer -i runs/'*'joggle_std=$joggle_std-'*'07-04-2021_13'*' -o runs/perovskites-sg_split=200-231-joggle_std=$joggle_std-joggle_rate=0.3
done
```

## Upload Tensorboard Runs

```sh
tensorboard dev upload --logdir jdft2d-sg_split-10runs-joggle --name "jdft exfoliation energy cgcnn 10-run-avrg atom joggling" --description "Joggling amplitudes = [0, 0.01, 0.02, 0.03] A, joggling rate = 0.3, test set = spacegroup 75-143"
```

## Tested Spacegroup Splits

- matbench_perovskites: 200-231 (221 is the only cubic spacegroup in that dataset, 1273 out of 18928 entries, just 6.7 , very small test set)
- matbench_jdf2d: 75-143
- log_g/kvrh: 1-100 (approx. 70/30 split)
- matbench_phonons: 1-100 (approx. 70/30 split)

## Resume Multiple with Fine Tuning

Trial run:

```sh
for joggle_std in 0 0.01 0.02 0.03
do
  python examples/cgcnn-sg-split.py --train --evaluate --log --data-name matbench_perovskites --targets 'e_form' --tasks regression --losses L1 --epochs 600 --joggle-std $joggle_std --joggle-rate 0.3 --sg-split 200-231 --fint-tune models/perovskites-xxl_cgcnn-sg_split=200-231-joggle_std=$joggle_std-joggle_rate=0.3/checkpoint-r0.pth.tar
done
```

Single `sbatch`:

```sh
for joggle_std in 0 0.01 0.02 0.03
do
  sbatch -J xxl-cgcnn-fine-tune -t 5:0:0  --array 0-9 --export CMD="python examples/cgcnn-sg-split.py --train --evaluate --log --data-name matbench_perovskites --targets 'e_form' --tasks regression --losses L1 --epochs 200 --joggle-std $joggle_std --joggle-rate 0.3 --sg-split 200-231 --fine-tune models/perovskites-xxl_cgcnn-sg_split=200-231-joggle_std=$joggle_std-joggle_rate=0.3/checkpoint-r\$SLURM_ARRAY_TASK_ID.pth.tar --model-name perovskites-sg_split=200-231-joggle_std=$joggle_std-joggle_rate=0.3 --run-id \$SLURM_ARRAY_TASK_ID --learning-rate 3e-5" hpc/gpu_submit
done
```
