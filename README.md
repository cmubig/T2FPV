# T2FPV

Predicting pedestrian motion is essential for developing
socially-aware robots that interact in a crowded
environment. While the natural visual perspective for a
social interaction setting is an egocentric view, the
majority of existing work in trajectory prediction therein
has been investigated purely in the top-down trajectory
space. To support first-person view trajectory prediction
research, we present T2FPV, a method for constructing
high-fidelity first-person view (FPV) datasets given a
real-world, top-down trajectory dataset; we showcase our
approach on the ETH/UCY pedestrian dataset to generate the
egocentric visual data of all interacting pedestrians,
creating the T2FPV-ETH dataset. In this setting,
FPV-specific errors arise due to imperfect detection and
tracking, occlusions, and field-of-view (FOV) limitations
of the camera. To address these errors, we propose CoFE, a
module that further refines the imputation of missing data
in an end-to-end manner with trajectory forecasting
algorithms. Our method reduces the impact of such FPV
errors on downstream prediction performance, decreasing
displacement error by more than 10% on average. To
facilitate research engagement, we release our T2FPV-ETH
dataset and software tools.


![CoFE Example](vis_out_final/sgnet_naomi_univ_batch42_agent19.gif)

## Dependency Installation
- Create a conda environment:
    - `conda create --name fpv python=3.7`
    - `conda install -c pytorch torchvision cudatoolkit=11.0 -c pytorch`
    - `cat requirements.txt | xargs -n 1 python -m pip install`
- In numpy/lib/format.py, ensure that pickle.dump has protocol=4

## Dataset Preparation

You can either download the raw data directly and follow the pre-processing steps below, or skip directly to downloading the already processed data:

### Raw Data
- Download the raw data from the following link: https://cmu.box.com/s/tij0yyo8ulqh1n7uane0pf3onj7ror7f 
- Extract the files into the `data/` folder:
    - `tar -xvf FPVDataset.tar.gz; mv FPVDataset data/. ; rm FPVDataset.tar.gz`
- Run `python multi_run.py --run-type data` to construct the processed cache files for all the tracklets, data loaders, etc.

### Pre-Processed Data (TODO: update for later, since pickles break...)
- Download `input_data.pkl` from here: https://cmu.box.com/s/d5t0yyirtjjkodgreiv6r48yby4mtywv
    - Move the file into the `data/` folder
- Download the processed cache files here: https://cmu.box.com/s/gzaop5av0zsteotmfzyiwu6hgckbukh8
- Extract the `npy` files into the `data/processed` folder:
    - `tar -xvf processed.tar.gz; mv processed/* data/processed/.; rm processed.tar.gz`

## Running Experiments
- To test the install, run:
    - `python multi_run.py --exp-config config/fpv_noisy/vrnn.json --run-type data`
(TODO)
