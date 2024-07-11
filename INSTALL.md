Basic Installation
==================
First, clone the main branch of the ili-summarizer repository onto your local machine.
```bash
git clone git@github.com:florpi/ili-summarizer.git
```
Next, create a fresh anaconda environment, preferably in Python 3.10
```bash
conda create -n summ python=3.10
conda activate summ
```
Next, install [`pmesh`](https://github.com/rainwoodman/pmesh). We use `pmesh` as the primary meshing operator for most summary statistics. It requires an MPI library (`openmpi` or `mpich`), `gcc`, `gsl`, `mpi4py`, and `cython<=0.29.33` to be installed. For example, the following commands install `pmesh` on anvil@Purdue.
```bash
module load openmpi/4.1.6 gsl/2.4 gcc/11.2.0
pip install mpi4py cython==0.29.33 --no-cache-dir
pip install pmesh
```
Install ili-summarizer and its dependencies. We provide various backends for calculating summary statistics, and you can install them with:
```bash
pip install -e ili-summarizer            # for data loading and general utilities
pip install -e ili-summarizer[pypower]   # for power spectra and marked power spectra
pip install -e ili-summarizer[pycorr]    # for two-point correlation function
pip install -e ili-summarizer[polybin]   # for bispectrum
pip install -e ili-summarizer[kymatio]   # for wavelet statistics
pip install -e ili-summarizer[corrfunc]  # for density split and VPF
# OR
pip install -e ili-summarizer[all]       # for all of the above
```
Finally, verify your installation.
```bash
python -c "import summarizer; print(summarizer.__version__)"
```
If you did not install all the backends in the previous step, this import will print a warning message indicating which backends are missing.
