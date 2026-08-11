# DAF-FlowNet

Implementation code for **“Unsupervised 4D Flow MRI velocity enhancement and unwrapping using divergence-free neural networks.”**

## Example data

Example files simulated with [feelmri](https://github.com/hmella/feelmri) can be downloaded from [here](https://huggingface.co/javierbz24/DAF-FlowNet/tree/main).

Place the example files in `data/`:

```text
data/
├── sim.h5
└── ref.h5
```

Each HDF5 file contains `u`, `v`, and `w` arrays with shape `(time, x, y, z)`, a 3D `mask`, and the attributes `spacing` and `dt`.

## Training

Fit the network to a selected timeframe or the complete sequence:

```powershell
python src/train.py --timeframe 3
python src/train.py --all-timeframes
```

Training results are organized by model and timeframe:

```text
results/
└── model1/
    ├── time_frame_3/
    │   ├── checkpoints/
    │   ├── errors/
    │   ├── plots/
    │   ├── model_final.pth
    │   └── training.csv
    └── time_frame_4/
```

## Evaluation

Evaluate one timeframe or the complete sequence:

```powershell
python src/evaluate.py --timeframe 3
python src/evaluate.py --all-timeframes
```

Single-timeframe predictions and metrics are saved in `results/model1/time_frame_3/evaluation/`. Combined outputs from `--all-timeframes` are saved in `results/model1/evaluation/`.

## Citation

If you use this implementation please cite

```bibtex
@article{bisbal2026unsupervised,
  title={Unsupervised 4D Flow MRI Velocity Enhancement and Unwrapping Using Divergence-Free Neural Networks},
  author={Bisbal, Javier and Sotelo, Julio and Mella, Hern{\'a}n and Odeback, Oliver Welin and Mura, Joaqu{\'\i}n and Marlevi, David and Matsuda, Junya and Iwata, Kotomi and Sekine, Tetsuro and Tejos, Cristian and others},
  journal={arXiv preprint arXiv:2604.00205},
  year={2026}
}
```
