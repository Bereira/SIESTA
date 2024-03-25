# `SIESTA`: Statistical matchIng between rEal and Synthetic sTellaR PopulAtions

The `SIESTA` code is a Python tool developed to statistically fit the global parameters of star clusters, i.e., their metallicities, ages, distances, color excesses, and binary fractions. The code is based on the comparison between the stellar distribution in the observed color-magnitude diagram (CMD) and distributions from synthetic populations. A complete description of the code can be found at this reference (_insert reference later_).

The current version of the code uses isochrones from the [PARSEC-COLIBRI database](http://stev.oapd.inaf.it/cgi-bin/cmd) (version 3.7) and is optimized for CMDs in the $V-I \times V$ bands. We plan on including more flexibility in the photometric system and stellar model choice in future versions of the code.

## Licensing and referencing `SIESTA`

`SIESTA` is available in the XXX license (more details in LICENSE). If you use SIESTA, we only ask that you cite the original paper:

```
Insert @bibtex here
```

## Setting up your environment

To run `SIESTA` you will need several Python libraries. For [Anaconda](https://anaconda.org/) users, we provide the `siesta.yml` file to set up an environment to run the code. You can do that by running, in a terminal with Anaconda activate:

```
conda env create -f siesta.yml
```

You can then activate the enviroment by running:

```
conda activate siesta
```

Alternatively, you can open the `siesta.yml` and download each library manually. 

## Downloading isochrones

For `SIESTA` to run you'll need a grid of PARSEC-COLIBRI isochrones, that can be downloaded directly from [this website](http://stev.oapd.inaf.it/cgi-bin/cmd). Alternatively, the [ezPADOVA-2](https://github.com/asteca/ezpadova-2) package can be used for downloading a large number of isochrones automatically. In the current version of the SIESTA code, the isochrones must be in the $UBVRI$ photometric system. 

After downloading the isochrones you can run `CreateIsochoneGrid.py` to store them as individual files with the proper naming convention. 

## Characterizing a star cluster

To characterize a given star cluster, start by running `Initialization.ipynb` IPython Notebook, where you'll define all the necessary inputs for performing the Markov Chain Monte Carlo (MCMC) sampling. 

After running the Notebook, you can start the MCMC sampling by running:

```
python RunMCMCsampling.py PROJECT_NAME
```

replacing `PROJECT_NAME` by the name of your current project (defined in the `Initialization.ipynb` Notebook). 

To check your results after (or during) the sampling process, run the `ChainAnalysis.ipynb` Notebook. Keep in mind that for running this Notebook without issues, your MCMC chain must have, at least, 200 iterations. 


