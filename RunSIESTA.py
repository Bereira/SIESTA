#jfsaflksdfalskfhlaksdjfhjkl


#Import custom modules
import sys ;  sys.path.append('scripts')
import SIESTAmodules
#Import usefull libraries
from pandas import DataFrame,read_csv
from multiprocessing import Pool
from  emcee import EnsembleSampler
from emcee.moves import DESnookerMove,DEMove,KDEMove,GaussianMove,StretchMove,WalkMove
from pandas import read_csv
from numpy import inf,mean,absolute,savetxt,concatenate,loadtxt,isfinite,log,nan_to_num,errstate
from numpy.random import randint
from sklearn.neighbors import NearestNeighbors
from os import environ
from scipy.stats import binned_statistic_2d


def logPosterior(parameters):
    #Get parameters
    raw_mh,raw_age,d,red,binf = parameters
    #Evaluate priors
    logPrior = 0 
    logPrior += Prior_Metallicity(raw_mh,inputs['Priors']['Metallicity']['Parameters']) 
    logPrior += Prior_Age(raw_age,inputs['Priors']['Age']['Parameters']) 
    logPrior += Prior_Distance(d,inputs['Priors']['Distance']['Parameters']) 
    logPrior += Prior_Reddening(red,inputs['Priors']['Reddening']['Parameters']) 
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
        SIESTAmodules.SyntheticPopulation.Generator(syntCMD,
                                                     isochrone.copy(), 
                                                     inputs['Initial_population_size'],
                                                     binf, 
                                                     inputs['Companion_min_mass_fraction'],
                                                     d,red,
                                                     inputs['Photometric_limit'],
                                                     inputs['Error_coefficients'], 
                                                     inputs['Completeness_Fermi'],
                                                     PopulationSamplesRN, PhotometricErrorRN)
        #Use the estimator to evaluate the population density
        SyntheticGrid = SIESTAmodules.Distribution.Evaluate(inputs['Edges_color'], inputs['Edges_magnitude'],
                                                             syntCMD['Vfilled']-syntCMD['Ifilled'], syntCMD['Vfilled'],
                                                             renorm_factor=inputs['Cluster_size']/len(syntCMD))
        #Likelihood
        logLikelihood = SIESTAmodules.MCMCsupport.LikelihoodCalculator(ClusterGrid,
                                                                         SyntheticGrid,
                                                                         inputs['Temperature'],WeightsGrid) 
    #Fork: if not in range
    else:
        #Negative infinite likelihood
        logLikelihood = 0
    #Return likelihood
    return logLikelihood + logPrior/inputs['Temperature']


# INITIALIZATION
#Get inputs and create MCMC backend
inputs,backend,autocorr_timeOLD, answer = SIESTAmodules.Initialization.Start()
#Index for isochrones
isochroneIndex,_,_ = SIESTAmodules.Initialization.GetIsochroneIndex(inputs['Grid_path'])
#Create random numbers for the synthetic populations
PopulationSamplesRN, PhotometricErrorRN = SIESTAmodules.Initialization.RandomNumberStarter(inputs['Initial_population_size'],inputs['Seed'])


#CLUSTER DATA
print('Working with cluster data...')
#Import cluster data
ClusterGrid = loadtxt( 'projects/{}/ClusterGrid.dat'.format(inputs['Project_name']))
cluster_size = inputs['Cluster_size']
#Import filtered CMD
clusterCMD = read_csv('projects/{}/FilteredCMD.dat'.format(inputs['Project_name']),sep=',')
#Import Weights
WeightsGrid = loadtxt( 'projects/{}/WeightsGrid.dat'.format(inputs['Project_name']))


'''
with errstate(divide='ignore', invalid='ignore'):
    membWeight =  binned_statistic_2d(clusterCMD['V-I'], clusterCMD['V'], clusterCMD['memb'],statistic='sum',
                                  bins=[inputs['Edges_color'], inputs['Edges_magnitude']]).statistic.T / binned_statistic_2d(clusterCMD['V-I'], clusterCMD['V'], clusterCMD['memb'],statistic='count',
                                  bins=[inputs['Edges_color'], inputs['Edges_magnitude']]).statistic.T 

membWeight = nan_to_num(membWeight,nan=1)
'''


#MCMC
#Warn when running a tempered chain
if inputs['Temperature'] >1 : print('WARNING! Running with a tempered likelihood')
print('MCMC calculations...')
#Initialize walkers
walkers_start_position = SIESTAmodules.MCMCsupport.WalkersStartPosition(inputs['Walkers_start'],answer,backend)  
#Get priors
Prior_Metallicity,Prior_Age,Prior_Distance,Prior_Reddening,Prior_BinFraction = SIESTAmodules.MCMCsupport.DefinePriors(inputs['Priors'])

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
            converged = all(tau * 27 < sampler.iteration)
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
            savetxt( 'projects/{}/autocorrelation.dat'.format(inputs['Project_name']),
                    concatenate([autocorr_timeOLD,autocorr_time]))
            if converged:
                break
            tauOLD = tau

            
        



                              
    



