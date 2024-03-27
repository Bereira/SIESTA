#Import custom modules
import sys ;  sys.path.append('modules')
import MCMCsampling
#Import usefull libraries
from pandas import DataFrame,read_csv
from multiprocessing import Pool
from  emcee import EnsembleSampler
from emcee.moves import StretchMove
from pandas import read_csv
from numpy import inf,mean,absolute,savetxt,loadtxt,isfinite
from os import environ


def logPosterior(parameters):
    #Get parameters
    raw_mh,raw_age,d,extpar,binf = parameters
    #Evaluate priors
    logPrior = 0 
    logPrior += Prior_Metallicity(raw_mh,inputs['Priors']['Metallicity']['Parameters']) 
    logPrior += Prior_Age(raw_age,inputs['Priors']['Age']['Parameters']) 
    logPrior += Prior_Distance(d,inputs['Priors']['Distance']['Parameters']) 
    logPrior += Prior_ExtPar(extpar,inputs['Priors']['ExtinctionPar']['Parameters']) 
    logPrior += Prior_BinFraction(binf,inputs['Priors']['BinFraction']['Parameters'])
    #Fork: finite prior
    if isfinite(logPrior):
        #Round metallicity and age (since the grid has discrete values)
        mh = round(raw_mh,2)
        age = round(raw_age,2)
        #Import isochrone
        isochrone = read_csv('{}/{}.dat'.format(inputs['Grid_path'],isochroneIndex[mh][age]).replace('\\','/'),sep=',')
        #Create the population
        syntCMD = DataFrame()
        MCMCsampling.SyntheticPopulation.Generator(syntCMD,
                                                    isochrone.copy(), 
                                                    inputs['Initial_population_size'],
                                                    binf, inputs['Companion_min_mass_fraction'],
                                                    d,extpar,inputs['ExtinctionLaw'],
                                                    inputs['IsochroneColumns'],inputs['Bands_Obs_to_Iso'],
                                                    inputs['Photometric_limit'],
                                                    inputs['Error_coefficients'], 
                                                    inputs['Completeness_Fermi'],
                                                    PopulationSamplesRN, PhotometricErrorRN)
        #Use the estimator to evaluate the population density
        SyntheticGrid = MCMCsampling.Distribution.Evaluate(inputs['Edges_color'], inputs['Edges_magnitude'],
                                                           syntCMD[SyntColl1Band]-syntCMD[SyntColl2Band], syntCMD[SyntMagBand],
                                                           renorm_factor=inputs['Cluster_size']/len(syntCMD))
        #Likelihood
        logLikelihood = MCMCsampling.MCMCsupport.LikelihoodCalculator(ClusterGrid, SyntheticGrid)
    #Fork: if not in range
    else:
        #Negative infinite likelihood
        logLikelihood = 0
    #Return likelihood
    return logLikelihood + logPrior


# INITIALIZATION
#Get inputs and create MCMC backend
inputs,backend = MCMCsampling.Initialization.Start()
#Index for isochrones
isochroneIndex,_,_ = MCMCsampling.Initialization.GetIsochroneIndex(inputs['Grid_path'])
#Create random numbers for the synthetic populations
PopulationSamplesRN, PhotometricErrorRN = MCMCsampling.Initialization.RandomNumberStarter(inputs['Initial_population_size'], inputs['ObsCatalogColumns'],inputs['Seed'])


#CLUSTER DATA
print('Working with cluster data...')
#Import cluster data
ClusterGrid = loadtxt( 'projects/{}/ClusterGrid.dat'.format(inputs['Project_name']))
cluster_size = inputs['Cluster_size']
#Import filtered CMD
clusterCMD = read_csv('projects/{}/FilteredCMD.dat'.format(inputs['Project_name']),sep=',')

#MCMC SAMPLING
print('MCMC calculations...')
#Initialize walkers
walkers_start_position = MCMCsampling.MCMCsupport.WalkersStartPosition(inputs['Walkers_start'])  
#Get priors
Prior_Metallicity,Prior_Age,Prior_Distance,Prior_ExtPar,Prior_BinFraction = MCMCsampling.MCMCsupport.DefinePriors(inputs['Priors'])

#DEFINE COLUMN NAMES
SyntMagBand = '{}_filled'.format(inputs['IsochroneColumns']['MagBand'] )
SyntColl1Band = '{}_filled'.format(inputs['IsochroneColumns']['ColorBand1'] )
SyntColl2Band = '{}_filled'.format(inputs['IsochroneColumns']['ColorBand2'] )
                                 
#Index and autocorrelation vector
index = 0
autocorr_time = []
tauOLD = inf


#Set number of threads to 1
environ["OMP_NUM_THREADS"] = "1"
#Check if it is the main process
if __name__ == '__main__':
#Start pool (for multiprocessing)
    with Pool() as pool:
        #Create sampler
        sampler = EnsembleSampler(len(walkers_start_position),
                                 5,
                                 logPosterior,
                                 backend=backend,
                                 moves = StretchMove(2),
                                 pool=pool)
        #Start samples
        for sample in sampler.sample(walkers_start_position, iterations=inputs['Max_iterations'] , progress=True):
            #Check for convergence every 100 steps
            if sampler.iteration %100: 
                continue
            #Autocorrelation time
            tau = sampler.get_autocorr_time(tol=0)
            autocorr_time += [ mean(tau) ] 
            index += 1
            #Acceptance fractioin
            af = sampler.acceptance_fraction
            #Convergence tests
            converged = all(tau * 50 < sampler.iteration)
            converged &= all(absolute(tauOLD - tau) / tau < 0.01)
            #Interact 
            i = sampler.iteration
            print('CONVERGENCE LOG #{}'.format(index))
            print('\t Iteration #{}'.format(i))
            print('\t Acceptance fraction: {}'.format(mean(af)))
            print('\t Autocorrelation times: {}'.format(tau))
            print('\t Increment: {}%'.format( (tau-tauOLD)/tau*100 ) )
            print('\t Chain size: {} autocorrelation times'.format(i/tau))
            #Save correlation time
            savetxt( 'projects/{}/autocorrelation.dat'.format(inputs['Project_name']),autocorr_time)
            if converged:
                break
            tauOLD = tau

            
        



                              
    



