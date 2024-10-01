# `SIESTA`: Statistical matchIng between rEal and Synthetic sTellar PopulAtions

The `SIESTA` code is a Python tool developed to statistically fit the global parameters of star clusters, i.e., their metallicities, ages, distances, color excesses, and binary fractions. The code is based on the comparison between the stellar distribution in the observed color-magnitude diagram (CMD) and distributions from synthetic populations. A complete description of the code can be found at this reference (_insert reference later_).

The current version of the code uses isochrones from the [PARSEC-COLIBRI database](http://stev.oapd.inaf.it/cgi-bin/cmd) (version 3.7). We plan on including more flexibility in the choice of stellar models in future versions of the code.

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

To run `SIESTA` properly you will need several Python libraries. For [Anaconda](https://anaconda.org/) users, we provide the `siesta.yml` file to set up an environment to run the code. You can do that by running, in a terminal with Anaconda activate:

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

Which will allow you to use the Notebooks in this enviroment by running:


```
jupyter notebook
```


## Downloading isochrones

For `SIESTA` to run you'll need a grid of PARSEC-COLIBRI isochrones, that can be downloaded directly from [this website](http://stev.oapd.inaf.it/cgi-bin/cmd). Alternatively, the [ezPADOVA-2](https://github.com/asteca/ezpadova-2) package can be used for downloading a large number of isochrones automatically. In the current version of the SIESTA code, the isochrones must be in the $UBVRI$ photometric system. 

After downloading the isochrones you can run `CreateIsochoneGrid.py` to store them as individual files with the proper naming convention. 

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

We provide, in this repository, a decontaminated catalog in the $VI$ bands for the cluster [Lindsay 113](http://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Lindsay+113&submit=SIMBAD+search), observed by the [VISCACHA](http://www.astro.iag.usp.br/~viscacha/), as well as a set of isochrones. These are meant to be used as examples: all Notebooks come filled so that you can simply run them for this cluster. You can also change some parameters to see what changes. Hopefully this will help you gain some intuition obn how to use `SIESTA` in your own work. Once you feel confident with your ability to run the code, feel free to delete the examples.
