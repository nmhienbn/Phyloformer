# PhyloFormer 2

Companion repo to the **"Likelihood-free inference of phylogenetic tree posterior distributions"** preprint:  
[arXiv:2510.12976](https://doi.org/10.48550/arXiv.2510.12976)  

There are submodules so don't forget the `--recursive` flag when cloning:
```
git clone --recursive git@gitlab.in2p3.fr:deelogeny/wp1/phyloformer-2.git
```

Install requirements
with `conda`:
```bash
conda create -n my_pf2_env python=3
conda install --yes --file requirements.txt
conda activate my_pf2_env

python path/to/pf2/script.py
```

with `uv`:
```bash
cd path/to/pf2/repo

uv init
uv add -r requirements.txt

uv run path/to/pf2/script.py
```

## Using PhyloFormer 2

### Inference with PF2
Use the [infer.py](./infer.py) script.

```bash
# Compute greedy MAP trees for a directory of example MSAs
uv run infer.py --mode max-sample ./data/tiny/msas/ ./pretrained/pf2.tch ./data/tiny/pred_MAPs/pf2/

# Sample 500 trees from the posterior for each example MSA
uv run infer.py --mode samples --nsamples 500 ./data/tiny/msas/ ./pretrained/pf2.tch ./data/tiny/posterior_samples/pf2/

```

Pretrained model weights are given in the [`pretrained` directory](./pretrained/):
 - `pf2.tch`: paper version of PF2 fine-tuned on multiple tree sizes from `pf2_base.tch`.
 - `pf2_base.tch`: base PF2 instance, trained on 50-taxa trees with LG+G8+indels MSAs.
 - `pf2_cherry.tch`: PF2 instance pretrained on LG+G8+indels MSAs and fine-tuned on CherryML MSAs.
 - `pf2_selreg.tch`: PF2 instance pretrained on LG+G8+indels MSAs and fine-tuned on SelReg MSAs.
 - `pf2_revbayes.tch`: PF2 instance pretrained on LG+G8+indels MSAs and fine-tuned on LG+G8+indels MSAs and trees from RevBayes' Uniform prior.
 - `pf2_potts.tch`: PF2 instance trained on MSAs generated under a Potts model.

### Training a PF2 Model
Use the [train.py](./train.py).

If you wish to train or fine-tune a PF2 instance you will need to add the additional 
training dependencies: 

```bash
# e.g.
uv add -r requirements.train.txt

uv run train.py ...
```

## Reproducing Figures
Coming soon!
