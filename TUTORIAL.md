# Tutorial

We assume all the commands below are run in a new empty folder.

Verify your Python version is at least 3.8.

```bash
python --version
```

Create a fresh virtual environment.
```bash
python -m venv .venv
```

Activate the fresh environment.
```bash
source .venv/bin/activate
```

Install RASSINE

```bash
pip install rassine
```

Get the scripts needed to run the RASSINE pipeline steps in parallel, and set the executable flag. You can skip the first two lines if GNU Parallel is already installed on your computer.

```bash
curl -L -O https://raw.githubusercontent.com/pegasilab/rassine/master/parallel
chmod +x parallel
curl -L -O https://raw.githubusercontent.com/pegasilab/rassine/master/run_rassine.sh
chmod +x run_rassine.sh
```

Download and unzip star data

```bash
curl -L -O https://github.com/pegasilab/HD110315/archive/refs/heads/master.zip
unzip master.zip
```

Create and populate the configuration file `harpn.ini`. See [the latest file on the RASSINE repository](https://raw.githubusercontent.com/pegasilab/rassine/master/harpn.ini) for updates.

```
[common]
pickle-protocol = 4

[run_rassine]
nprocesses=4
nchunks=10
nice=18

[preprocess_import]
instrument = HARPN
drs-style=new
plx-mas=0.0

[reinterpolate]
dlambda = 0.01

[stacking_create_groups]
bin-length=1.0
dbin=0.0

[rassine]
par-stretching=auto_0.5
par-vicinity=7
par-smoothing-box=6
par-smoothing-kernel=savgol
par-fwhm=auto
CCF-mask=master
par-R=auto
par-Rmax=auto
par-reg-nu=poly_1.0
mask-telluric=[[6275,6330],[6470,6577],[6866,8000]]
synthetic-spectrum=false
interpolation=cubic
denoising-dist=5
count-cut-lim=3
count-out-lim=1
random-seed=

[matching_anchors_scan]
copies-master=0  
fraction=0.2 
threshold=0.66 
tolerance=0.5 

[matching_diff]
savgol-window=200
```

Run the pipeline.
```bash
./run_rassine.sh -c harpn.ini HD110315-master/data/s1d/HARPN
```
