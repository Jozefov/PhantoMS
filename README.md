
<p align="center">
  <img src="assets/PhantoMS_logo.png" width="90%"/>
</p>

## 🧠 PhantoMS

**PhantoMS** is a suite of graph‑neural‑network models and analysis tools designed for multi‑stage mass spectrometry. Built on the MassSpecGymMSn benchmark, PhantoMS implements both the standard and bonus variants of the Retrieval and De novo challenges. By training on progressively deeper fragmentation stages, our models learn richer internal representations, and achieved significant improvements over MS/MS baselines. Overall it focuses on:
- 🧪 **Multi‑stage MSn mass spectra**: handle MSn mass spectra 
- 🔍 **Retrieval** & ✨ **Bonus retrieval**: rank candidate molecules (with or without ground‑truth formulae)
- 🔍**De novo generation** & ✨ **Bonus De novo**: predict molecular graphs, optionally conditioned on true formulae
- 🧠**Centered Kernel Alignment (CKA)**: investigation of internal models representations trained on different MSn fragmentation stages 


## 📦 Installation

```bash
git clone git@github.com:Jozefov/PhantoMS.git
cd PhantoMS
conda env create -f environment.yml
conda activate phantoms_env
```

## ⚙️ Retrieval Example

Below is a minimal example of a GNN‑based retrieval model using the MassSpecGymMSn data loader:
```python
# Imports
from massspecgym.data.datasets import MSnRetrievalDataset
from massspecgym.data.transforms import MolFingerprinter
from massspecgym.featurize import SpectrumFeaturizer
from massspecgym.data import MassSpecDataModule
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch
from pytorch_lightning import Trainer

# 1. Data setup (paths match MSn benchmark)
mass_spectra     = "20241211_msn_library_pos_all_lib_MSn.mgf"
candidates_json  = "MassSpecGymMSn_retrieval_candidates_mass.json"
candidates_cache = "MassSpecGymMSn_retrieval_candidates_mass.h5"
split_file       = "20241211_split.tsv"

config = {
    'features': ['collision_energy', 'ionmode', 'adduct', 'spectrum_stats', 
                 'atom_counts', 'value', "retention_time", 'ion_source',
                 'binned_peaks'],
    'feature_attributes': {
        'atom_counts': {
            'top_n_atoms': 12,
            'include_other': True,
        },
    },
}
featurizer = SpectrumFeaturizer(config, mode='torch')

fp_size = 2048
batch_size = 12

dataset = MSnRetrievalDataset(
    pth=mass_spectra,
    candidates_pth=candidates_json,
    featurizer=featurizer,
    cache_pth=candidates_cache,
    mol_transform=MolFingerprinter(fp_size=fp_size),
)
data_module = MassSpecDataModule(
    dataset=dataset,
    batch_size=batch_size,
    split_pth=split_file,
    num_workers=4,
)
data_module.prepare_data()
data_module.setup()

# 2. Model definition
class GNNRetrievalModel(RetrievalMassSpecGymModel):
    def __init__(self, hidden_channels=128, out_channels=fp_size,
                 node_feature_dim=1039, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Two-layer GCN
        self.conv1 = GCNConv(node_feature_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Fingerprint head
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

    def step(self, batch, stage):
        data = batch['spec']
        fp_true = batch['mol']
        cands = batch['candidates']
        ptr = batch['batch_ptr']
        fp_pred = self.forward(data)
        loss = nn.functional.mse_loss(fp_pred, fp_true)
        # repeat and cosine-sim for retrieval scores
        pred_rep = fp_pred.repeat_interleave(ptr, dim=0)
        scores = nn.functional.cosine_similarity(pred_rep, cands)
        return {'loss': loss, 'scores': scores}

# 3. Train & test
model = GNNRetrievalModel(hidden_channels=128,
                          out_channels=fp_size,
                          node_feature_dim=1039,
                          at_ks=(1,5,20))
trainer = Trainer(accelerator="cpu", devices=1,
                  max_epochs=2,
                  limit_train_batches=2,
                  limit_val_batches=2)
trainer.fit(model, data_module)
trainer.test(model, data_module)
```

## 🔗 References

If you use PhantoMS, or MassSpecGymMSn in your work, please cite the following paper:

```bibtex
@mastersthesis{jozefov2025massspecgymmsn,
  title={Predicting molecular structures from multi-stage MSn fragmentation trees using graph neural networks and DreaMS foundation model},
  author={Jozefov, Filip},
  year={2025},
  school={Faculty of Informatics, Masaryk University}
}
```

For the original MassSpecGym benchmark:

```bibtex
@article{bushuiev2024massspecgym,
      title={MassSpecGym: A benchmark for the discovery and identification of molecules}, 
      author={Roman Bushuiev and Anton Bushuiev and Niek F. de Jonge and Adamo Young and Fleming Kretschmer and Raman Samusevich and Janne Heirman and Fei Wang and Luke Zhang and Kai Dührkop and Marcus Ludwig and Nils A. Haupt and Apurva Kalia and Corinna Brungs and Robin Schmid and Russell Greiner and Bo Wang and David S. Wishart and Li-Ping Liu and Juho Rousu and Wout Bittremieux and Hannes Rost and Tytus D. Mak and Soha Hassoun and Florian Huber and Justin J. J. van der Hooft and Michael A. Stravs and Sebastian Böcker and Josef Sivic and Tomáš Pluskal},
      year={2024},
      eprint={2410.23326},
      url={https://arxiv.org/abs/2410.23326},
      doi={10.48550/arXiv.2410.23326}
}
```