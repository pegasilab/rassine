

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
Download the default configuration file for HARPN.

```bash
curl -L -O https://raw.githubusercontent.com/pegasilab/rassine/master/harpn.ini
```

Run the pipeline. You can look at the `HD110315-master/data/s1d/HARPN/STACKED/` directory to
inspect progress, especially during the "rassine" normalization stage

```bash
./run_rassine.sh -c harpn.ini HD110315-master/data/s1d/HARPN
```
