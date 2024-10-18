# `SIESTA`: Statistical matchIng between rEal and Synthetic sTellar PopulAtions

The `SIESTA` code is a Python tool developed to statistically fit the global parameters of star clusters, i.e., their metallicities, ages, distances, color excesses, and binary fractions. The code is based on the comparison between the stellar distribution in the observed color-magnitude diagram (CMD) and distributions from synthetic populations. It allows for a flexible choice of photometric system and stellar evolution models.

A complete description of the code can be found at [this paper](https://academic.oup.com/mnras/article/533/4/4210/7746770)

## Using and referencing `SIESTA`

`SIESTA` is available in the GNU General Public License, version 3.0 (more details in the LICENSE file). If you use SIESTA, we kindly ask that you cite the original paper:

```
@article{10.1093/mnras/stae2055,
    author = {Ferreira, Bernardo P L and Jr., João F C Santos and Dias, Bruno and Maia, Francisco F S and Kerber, Leandro O and Gardin, João Francisco and Oliveira, Raphael A P and Westera, Pieter and Rocha, João Pedro S and Souza, Stefano O and Hernandez-Jimenez, Jose A and Santrich, Orlando Katime and Villegas, Angeles Pérez and Garro, Elisa R and Baume, Gustavo L and Fernández-Trincado, José G and de Bórtoli, Bruno and Parisi, Maria Celeste and Bica, Eduardo},
    title = "{The VISCACHA Survey – XI. Benchmarking SIESTA: a new synthetic CMD fitting code}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {533},
    number = {4},
    pages = {4210-4233},
    year = {2024},
    month = {08},
    issn = {0035-8711},
    doi = {10.1093/mnras/stae2055},
    url = {https://doi.org/10.1093/mnras/stae2055},
    eprint = {https://academic.oup.com/mnras/article-pdf/533/4/4210/59075798/stae2055.pdf},
}
```

## Setting up your environment

To run `SIESTA` properly you will need several Python libraries. The `siesta.yml` file can be used to set up an environment for running the code. For [Anaconda](https://anaconda.org/) users, you can create the environment by running, in a terminal with Anaconda activated:

```
conda env create -f siesta.yml
```

You can then activate the environment by running:

```
conda activate siesta
```

Alternatively, you can open the `siesta.yml` and download each library manually. 

You will also need to be able to run [Jupyter Notebooks](https://jupyter.org/) in your computer. If you don't have it already installed, you can download it by running, with the `siesta` enviroment activated:

```
conda install jupyter
```

This will allow you to use the Notebooks in this environment by running:

```
jupyter notebook
```


## Downloading isochrones

For `SIESTA` to run you'll need a grid of isochrones with uniform spacing in $\log Age$ and $[M/H]$. The isochrones need to be stored individually as `.dat`, files with comma-separated columns, and the first row corresponding to the labels, which must be consistent throughout the grid. You'll also need a `.pkl` file that correlates each file and the corresponding age/metallicity pair. These files can be generated from a *Python* [```defaultdict```](https://docs.python.org/3/library/collections.html#collections.defaultdict) with the structure:

```
index[metallicity][logAge] = Corresponding file name
```

For PARSEC-COLIBRI isochrones, you can download them directly from [this website](http://stev.oapd.inaf.it/cgi-bin/cmd). Alternatively, the [ezPADOVA-2](https://github.com/asteca/ezpadova-2) package can be used for downloading a large number of isochrones automatically. Similarly, MIST isochrones can be downloaded manually from [this website](https://waps.cfa.harvard.edu/MIST/interp_isos.html#), and the [`ezMIST`](https://github.com/mfouesneau/ezmist) package can be used for automatic download.

If you use either of these packages, you can run the Notebook `CreateIsochoneGrid.ipnb` to store them as individual files with the proper naming conventionon to execute `SIESTA`. The Notebook is compatible with PARSEC-COLIBRI isochrones version 3.7 + eZPADOVA-2 with the last comment on Aug 17, 2023, and MIST isochrones version 1.2 + eZMIST with the last commit on Dec 10, 2020

For other stellar evolution models (e.g., BASTI, YY), although `SIESTA` should be able to use them for statistical fitting, there are no public, automatic downloading tools, to the best of my knowledge. 

## Characterizing a star cluster

To characterize a given star cluster, start by running `Initialization.ipynb` Jupyter Notebook, where you'll define all the necessary inputs for performing the Markov Chain Monte Carlo (MCMC) sampling. 

After running the Notebook, you can perform the MCMC sampling by running:

```
python RunMCMCsampling.py PROJECT_NAME
```

replacing `PROJECT_NAME` with the name of your current project (defined in the `Initialization.ipynb` Notebook). 

To check your results after (or during) the sampling process, run the `ChainAnalysis.ipynb` Notebook. Keep in mind that for running this Notebook without issues, your MCMC chain must have, at least, 200 iterations. 

### Important warning for Windows users

The `SIESTA` code uses the [``multiprocessing``](https://docs.python.org/3/library/multiprocessing.html) package for parallelism during the MCMC sampling. The current implementation of the code is not stable on Windows machines. While running the Notebooks is perfectly safe on Windows, running the MCMCsampling will likely lead to crashes. If you intend on executing SIESTA on a Windows machine, we recommend using [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).


## Example

We provide, in this repository, a decontaminated catalog in the $VI$ bands for the cluster [Lindsay 113](http://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Lindsay+113&submit=SIMBAD+search), observed by the [VISCACHA](http://www.astro.iag.usp.br/~viscacha/), as well as a small set of PARSEC-COLIBRI isochrones, with filters and an age and metallicity interval chosen specifically for the analysis of this cluster (for using `SIESTA` in other contexts, you'll probably need to download more isochrones). These are meant to be used as examples: all Notebooks come filled so that you can simply run them for this cluster. You can also change some parameters to see what changes. Hopefully this will help you gain some intuition obn how to use `SIESTA` in your own work. Once you feel confident with your ability to run the code, feel free to delete the examples.
