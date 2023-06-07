# Installation Instructions


## Basic installation
First, clone the main branch of the ili-summarizer repository onto your local machine.
```bash
    git clone git@github.com:florpi/ili-summarizer.git
```
Next, install ili-summarizer from this repository using pip with the editable flag. This ensures that any changes to the ili-summarizer directory are actively reflected in your python kernels.
```bash
    pip install -e 'ili-summarizer[backends]'
```

## Verify installation

TODO

## Alternative installations
ili-summarizer consolidates the usage of [several backend packages](setup.cfg#L16) to calculate various summary statistics. Dependent on your development environment, dependancy conflicts of these packages may be difficult to resolve (the most problematic is [nbodykit](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html#conda-installation)). We provide example installations with specific configurations which install. Experience on different systems may vary...

### Without backends
In the case where you only want to use ili-summarizer for its dataloading functionality (e.g. as a dependency of ltu-ili), we've provided an installation option which does not install any conflicting backends. Instead of the `pip install` command in the Basic Installation, use:
```bash
    pip install -e ili-summarizer
```
This will allow you to use ili-summarizer to load previously-calculated summary statistics but will prevent you from calcuating any new statistics with each respective backend. However, you can also use this installation option as a blank canvas to install a subset of the backends which you find necessary for your work.

### MacOS Monterey 12.6, Macbook Air, Apple M2
TODO

### Rocky Linux 8.7 / Redhat (infinity@IAP)
TODO