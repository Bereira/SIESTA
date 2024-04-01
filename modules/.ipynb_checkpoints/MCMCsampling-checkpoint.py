class Auxiliary:
    #Miscellaneous functions useful in different parts of the code

    def Round_to_Reference(x,xref):
        # 
        # ROUND VALUE TO LIST OF REFERENCES
        #
        # INPUTS
        #   x: float to be rounded
        #   xref: array of reference values
        #
        # OUTPUTS
        #   Rounded value of x, to the closest value in xref
        #
        #Import specific libraries
        from numpy import argmin
        #Return
        return xref[ argmin((x-xref)**2) ]


class Initialization:
    #Functions for importing data or creating object for future use in the program
    
    def Start():
        #
        # IMPORT INPUTS AND PREPARE FILES
        #
        # Loads input file and creates other files for the MCMC sampling
        #
        # OUTPUTS
        #   inputs: dictionary containing user defined inputs
        #   backend: backend to store MCMC chain
        #
        #Import libraries
        from sys import argv,exit
        from os.path import exists
        from os import remove
        from pickle import load
        from emcee.backends import HDFBackend
        from sys import exit
        #Receive parameters from terminal
        project_name = argv[1]
        #Set path
        path = 'projects/{}'.format(project_name)
        #Import inputs dictionary
        with open('{}/inputs.pkl'.format(path),'rb') as pkl_file: inputs = load(pkl_file)
        #Add project name to inputs
        inputs['Project_name'] = project_name
        #
        # MCMC backend
        #
        #Backend file
        backend_file = '{}/backend.h5'.format(path)
        #Check if back end exists
        if exists(backend_file):
            #Warn the user about an existing backeng file
            #Question if we want to delete it
            print('A MCMC sampler already exists for this project...')
            print('\t 1 Delete it and start a new chain')
            print('\t 2 Abort MCMC sampling')
            answer = int(input('Answer: '))
            #Check if we want to delete it
            if answer == 1:
                #Delete backend file
                remove(backend_file)
                #Create a new   one
                backend = HDFBackend(backend_file)
            elif answer == 2:
                exit()
            else: 
                print('Choose a valid option!')
                exit()
        else: 
            backend = HDFBackend(backend_file)
        #Return parameters
        return inputs, backend
    
    def GetIsochroneIndex(path_to_grid_data):
        #
        # IMPORT ISOCHRONE INDEX
        #
        # Imports the index that relates ages and metallicities to individual files
        #
        # INPUTS
        #   path_to_grid_data: string containg the path towars a pickle file containing the isochrone grid
        #
        # OUTPUTS
        #   index: path to the isochrones  grid[metalicity][age] = path
        #   mh_list: list of available metalicities
        #   age_list: list of available ages (in log10 form)
        #
        # Import libraries
        from pickle import load
        from numpy import array
        #Open file
        with open('{}/index.pkl'.format(path_to_grid_data),'rb') as pkl_file:
            #Read grid
            index = load(pkl_file)
        #List of metallicities and ages
        mh_list = sorted(list(index.keys()))
        age_list = sorted(list(index[mh_list[0]].keys()))
        #New grid
        return index, array(mh_list), array(age_list)
    
    def RandomNumberStarter(synt_pop_size,ObsBands,seed=1):
        #
        # CREATE RANDOM NUMBERS
        #
        # Generate pseudomnumbers for producing synthetic CMDS
        #
        # INPUTS
        #   synt_pop_size: initial size for the synthetic populations
        #   ObsBands: dictionary with labels to the Observational bands
        #   Seed: integer seed for the random number generator (degault=1)
        #
        # OUTPUTS
        #   PopulationSamplesRN: array of uniformly generated random numbers between 0 and 1
        #       - Used for sampling initial masses for single and companion stars + completeness removal
        #   PhotometricErrorRN: array of random numbers generated using a normal distribution (avg = 0, std = 1)
        #       - Used for adding noise to the synthetic magnitudes
        #
        #Import libraries
        from numpy.random import RandomState
        from scipy.stats import truncnorm
        #Create random state from given seed
        RNG = RandomState(seed)
        #Sample uniform random numbers
        PopulationSamplesRN = RNG.random([synt_pop_size,3])
        #Truncated normal distribution function
        def truncated_normal(mean,std,low,up):
            return truncnorm( (low-mean)/std, (up-mean)/std, loc=mean, scale=std)
        #Sample Gaussian random numbers
        PhotometricErrorRN = [truncated_normal(mean=0,std=1,low=-1,up=1).rvs([synt_pop_size,3],random_state=seed),
                              truncated_normal(mean=0,std=1,low=-3,up=3).rvs([synt_pop_size,3],random_state=seed+1)]
        #Check if there are repetition in bands
        if ObsBands['MagBand'] == ObsBands['ColorBand1']:
            PhotometricErrorRN[0][:,1] = PhotometricErrorRN[0][:,0]
            PhotometricErrorRN[1][:,1] = PhotometricErrorRN[1][:,0]
        if  ObsBands['MagBand'] == ObsBands['ColorBand2']:
            PhotometricErrorRN[0][:,2] = PhotometricErrorRN[0][:,0]
            PhotometricErrorRN[1][:,2] = PhotometricErrorRN[1][:,0]

        #Return
        return PopulationSamplesRN, PhotometricErrorRN

        
class Distribution:
    #Functions for analyzing the distribution of points in the CMD
        
    def Evaluate(edges_col, edges_mag, CMDcolor, CMDmag, renorm_factor=1):
        #
        # EVALUATE THE DISTRUBUTION OF A POPULATION IN A CMD
        #
        # Given a grid of renormalized colors and magnitudes, count the number of CMD stars in a 2D histogram
        #
        # INPUTS
        #   edges_col, edges_mag: edges of the bins in color and magnitude
        #   CMDcolor,CMDmag: color and magnitude values of the CMD
        #   CMDgrid: pandas DataFrame containing all the (v-i,v) positions to evaluate the density
        #   renorm_factor: renormalization factor for the counts (default=1)
        #
        # OUTPUTS
        #   The Hess diagram as a 2D histogram
        #
        #Import libraries
        from numpy import histogram2d
        #Evaluate histogram
        CMDhist,_,_ = histogram2d(CMDcolor,CMDmag,bins=[edges_col,edges_mag])
        #Return
        return CMDhist.T*renorm_factor
                                  
        
class SyntheticPopulation:
    # Functions for creating a synthetic CMD from a given isochrone
    
    def IsochroneAnalyzer(isochrone,phot_lim,d,extpar,extlaw,Nwanted,binary_fraction,IsoBands):
        #
        # EXTRACT IMPORTANT INFORMATION ABOUT THE ISOCHRONE
        #
        # Collects several information about the isochrone, that will be useful in the generation of the synthetic CMD
        #
        # INPUTS
        #   isochrone: pandas DataFrame containing a PARSEC-COLIBRI isochrone
        #   phot_limit: fainter possible magnitude for the synthetic population
        #   d: distance (in kpc) of the population (for evaluation apparent magnitudes and errors)
        #   extpar: extinction parameter
        #   extlaw: dictionary relating the extinction parameter with the extinction coefficient and the color excess
        #   Nwanted: size of the population (single stars + non-resolved binaries) before completeness removal
        #   binary_fraction: fraction of the synthetic population to be composed of non-resolved binaries
        #   IsoBands: list with the band names in the isochrone files
        #
        # OUTPUTS
        #   Nsolo: number of single stars
        #   Nbin: Number of companions
        #   m_start: minimum mass of the isochrone (above the photometric limit)
        #   m_end: maximum mass of the isochrone
        #   
        #
        # Import library
        from numpy import array,log10
        #Get names
        magIso,col1Iso,col2Iso = IsoBands
        #Displace de isochrone according to the distance and extinction parameter
        isochrone['App'+magIso] = isochrone[magIso] + 5*log10(d*1000) - 5 + extpar*extlaw['MagCorrection']
        isochrone['App'+col1Iso] = isochrone[col1Iso] + 5*log10(d*1000) - 5 + extpar*extlaw['ColorCorrection1']
        isochrone['App'+col2Iso] = isochrone[col2Iso] + 5*log10(d*1000) - 5 + extpar*extlaw['ColorCorrection2']
        #Filter isochrone
        isochrone_brighter = isochrone[ isochrone['App'+magIso] <= phot_lim + 1 ].reset_index(drop=True)
        # Number of binary systems stars
        Nbin = int( Nwanted * binary_fraction)
        # Number of single stars
        Nsolo = Nwanted - Nbin
        # List with the minimum possible masses
        m_start = array( [isochrone_brighter['Mini'].min()] * Nwanted )
        #List with the maximum possible masses
        m_end = array( [ isochrone_brighter['Mini'].max()] * Nwanted )
        #Return
        return Nsolo, Nbin, m_start, m_end

    def KroupaMassSampling(m_inf,m_sup,random_numbers,N):
        #
        # SAMPLES MASS USING KROUPA (2001) IMF
        #
        # From lists of minimum and maximum masses, samples, for each element, a random mass using the Kroupa IMF as a PDF
        #   - This is achieved by mapping the PDF into another, uniform between the interval [0,1] and zero otherwise
        #   - Several regimes of the starting and final mass are considered because the Kroupa IMF is discontinuos at 0.5Msun
        #
        # INPUTS
        #   m_inf: 1D array containing the mininum possible mass for each element 
        #   m_sup: 1D array containing the maximum possible mass for each element 
        #   random_numbers: 1D array containing random uniformly generated numbers between 0 and 1
        #   N: number of elements to be sampled 
        #
        # OUTPUTS
        #   masses: 1D array containing the sampled masses
        #
        #Import libraries
        from numpy import zeros,where
        #Sample pop. where the upper limit is smaller than 0.5Msun
        def KroupaSmallMass(m_inf,m_sup,random_numbers):
            #Useful constants
            c1 = m_inf**-0.3
            c2 = m_sup**-0.3
            c3 = c1-c2
            #Calculating masses
            return (c1 - c3*random_numbers )**(-1/0.3)
        #Sample pop. where the lower limit is larger than 0.5Msun
        def KroupaLargeMass(m_inf,m_sup,random_numbers):
            #Useful constants
            c1 = m_inf**-1.3
            c2 = m_sup**-1.3
            c3 = c1-c2
            #Calculating masses
            return ( c1 - c3*random_numbers )**(-1/1.3)
        #Sample pop. where the lower limit is smaller than 0.5Msun and the upper limit is larger tha 0.5Msun
        def KroupaInterMass(m_inf,m_sup,random_numbers):
            #Empty array for results:
            masses = zeros(len(random_numbers))
            #Useful constants
            c1 = 0.5**-0.3
            c2 = 0.5**-1.3
            c3 = m_inf**-0.3
            c4 = m_sup**-1.3
            c5 = (c3 - c1)/0.3
            c6 = 0.5*(c2 - c4)/1.3
            #IMF normalization constant
            norm = 1/(c5+c6) 
            #Identifying the random number that corresponds to the mass where the IMF breaks
            break_num = norm*c5
            #Identifying mass regimes (break point of IMF)
            idx = random_numbers <= break_num
            #Sampling masses
            masses[idx] = ( c3[idx] - 0.3 * random_numbers[idx]/norm[idx] )**(-1/0.3)
            masses[~idx] = ( c2 + 1.3/0.5*( c5[~idx] - random_numbers[~idx]/norm[~idx] ) )**(-1/1.3)    
            return masses
        #Empty array for results
        masses = zeros(N) 
        #Sample for each case
        masses = where( m_sup<=0.5, KroupaSmallMass(m_inf,m_sup,random_numbers), masses )
        masses = where( m_inf>=0.5, KroupaLargeMass(m_inf,m_sup,random_numbers), masses )
        masses = where( (m_inf<0.5)&(m_sup>0.5),KroupaInterMass(m_inf,m_sup,random_numbers), masses  )
        #Return
        return masses 

    def InterpolateStars(isochrone,sampled_masses,IsoBands):
        #
        # SAMPLE STARS
        #
        # Interpolates a group of sampled masses from the isochrone
        #
        # INPUTS
        #   isochrone: pandas DataFrame containg the PARSEC isochrone
        #   sampled_masses: 1D array containg the masses that we want to interpolate from the isochrone
        #   IsoBands: list with the band names in the isochrone files
        #
        # OUTPUTS
        #   mag,col1,col2: 1D array containing the sampled magnitudes
        #   mass:  1D array containing the interpolated masses
        #
        #Import libraries
        from numpy import interp
        #Get names
        magIso,col1Iso,col2Iso = IsoBands
        #Interpolate CMD
        mag = interp(sampled_masses, isochrone['Mini'], isochrone['App'+magIso])
        col1 = interp(sampled_masses, isochrone['Mini'], isochrone['App'+col1Iso])
        col2 = interp(sampled_masses, isochrone['Mini'], isochrone['App'+col2Iso])
        mass = interp(sampled_masses, isochrone['Mini'], isochrone['Mass'])
        #Return
        return mag,col1,col2,mass
  
    def SampleBinaries(syntPop,isochrone,companion_minimum_mass_fraction,Nwanted,Nsolo,Nbin,BinarySamplesRN,IsoBands):
        #
        # SAMPLE BINARIES
        #
        # Interpolates a group of sampled masses from the isochrone to create companion stars, then append their fluxes
        #
        # INPUTS
        #   syntPop: empty pandas DataFrame, to store the synthetic population
        #   isochrone: pandas DataFrame containg the PARSEC isochrone,
        #   companion_minimum_mass_fraction: minimum mass fraction for companion in binary systems
        #   sampled_masses: 1D array containg the masses that we want to interpolate from the isochrone
        #   Nwanted: size of the population (single stars + non-resolved binaries) before completeness removal
        #   Nsolo: number of single stars
        #   Nbin: Number of companions
        #   BinarySamplesRN: random numbers for sampling the companions (uniformly generated between 0 and 1)
        #   IsoBands: list with the band names in the isochrone files
        #
        # OUTPUTS
        #   Mini_companion: companion initial mass
        #   Mag_companion,Col1_companion,Col2_companion: companion magnitudes
        #   Mass_companion: companion mass
        #   MagTotal,Col1Total,Col2Total: final magnitudes
        #
        # Import libraries
        from numpy import zeros, full, nan
        #Function for summing magnitudes
        def SumMagnitudes(mag1,mag2):
            #Import specific libraries
            from numpy import log10
            #Return
            return -2.5*log10( 10**(-0.4*mag1) + 10**(-0.4*mag2) )
        #Get names
        magIso,col1Iso,col2Iso = IsoBands
        # Empty arrays
        Mini_companion = zeros(Nwanted)
        Mass_companion = zeros(Nwanted)
        Mag_companion = full(Nwanted,nan)
        Col1_companion = full(Nwanted,nan)
        Col2_companion = full(Nwanted,nan)
        # Select the part of the CMD that will be formed of non-resolved binary systems
        binaryPop = syntPop[syntPop['IsBinary']]
        #Inferior mass limit
        m_start = zeros(Nbin) + binaryPop['Mini_single'] * companion_minimum_mass_fraction
        #Superior mass limit
        m_end = zeros(Nbin) + binaryPop['Mini_single']
        #Sample the masses of the population
        Mini_companion[Nsolo:] = BinarySamplesRN[Nsolo:] * (m_end - m_start) + m_start
        #Interpolate the population
        Mag_companion[Nsolo:],Col1_companion[Nsolo:],Col2_companion[Nsolo:],Mass_companion[Nsolo:] = \
            SyntheticPopulation.InterpolateStars(isochrone,Mini_companion[Nsolo:],IsoBands)
        # Total magnitudes: starts with the single stars, than combines the magnitudes of the binaries
        MagTotal = syntPop['App{}_single'.format(magIso)].copy()
        Col1Total = syntPop['App{}_single'.format(col1Iso)].copy()
        Col2Total = syntPop['App{}_single'.format(col2Iso)].copy()
        MagTotal[Nsolo:] = SumMagnitudes(binaryPop['App{}_single'.format(magIso)],Mag_companion[Nsolo:])
        Col1Total[Nsolo:] = SumMagnitudes(binaryPop['App{}_single'.format(col1Iso)],Col1_companion[Nsolo:])
        Col2Total[Nsolo:] = SumMagnitudes(binaryPop['App{}_single'.format(col2Iso)],Col2_companion[Nsolo:])
        #Return
        return Mini_companion,Mag_companion,Col1_companion,Col2_companion,Mass_companion,MagTotal,Col1Total,Col2Total
    
    def CompletenessRemovalFermi(syntPop,CompInfo,random_numbers,Obs_to_Iso_Bands):
        #
        # REMOVE STARS BASED COM COMPLETENESS CRITERIUM
        #
        # INPUTS
        #   syntPop: empty pandas DataFrame, to store the synthetic population
        #   CompInfo: dictionary withFermi function parameters, describing completeness: [FermiMag,Beta,Band]
        #   Obs_to_Iso_Bands: dictionary relating the previous band names
        #   random_numbers: 1D array containing random uniformly generated numbers between 0 and 1 
        #
        # OUTPUTS
        #   incomplete_size: size of the incomplete population
        #
        #Import libraries
        from numpy import exp
        # Define Fermi function
        def FermiFunction(x,xf,beta):
            return 1 / (1+exp( beta*(x-xf)  ))
        #Define band
        band = Obs_to_Iso_Bands[CompInfo['Band']]
        #Evaluate completeness for synthetic population
        syntPop['Completeness'] = FermiFunction(syntPop['App{}'.format(band)],CompInfo['FermiMag'],CompInfo['Beta'])
        #Drop stars randomly, proportionaly to the completeness
        syntPop.drop( syntPop[ random_numbers > syntPop['Completeness']].index,inplace=True )
        #Return new size
        return len(syntPop)
    
    def AddPhotometricUncertainty(mag,col1,col2,error_coeffs,PhotometricErrorRN,incomplete_size):
        # ADD NOISE TO DATA
        #
        # Add noise to the population magnitudes, using the expected photometric error
        #
        # INPUTS
        #   mag,col1,col2: magnitudes of the synthetic population members
        #   error_coeffs: dictionary containing coefficients for an exponential fit of the photometric errors
        #   RNG: numpy RandomState
        #   incomplete_size: size of the synthetic CMD after completeness removal
        #
        # OUTPUTS
        #   MagObs, Col1Obs, Col2Obs: magnitudes added with noise
        #   Magfilled, Col1filled, Col2filled: magnitudes added with noise a second time (to emulate filling performed with observational data)
        #   noiseMag, noiseCol1, noiseCol2: noise added to each magnitude
        #   noiseMagfilled, noiseCol1filled, noiseCol2filled: second noise added to each magnitude
        #
        #Define exponential function
        def Exponential(x,a,b,c):
            #Import specific libraries
            from numpy import exp
            #Return
            return a * exp( b * x ) + c
        
        #Evaluate errors
        errorMag =  Exponential(mag,error_coeffs['MagBand'][0],error_coeffs['MagBand'][1],error_coeffs['MagBand'][2])
        errorCol1 =  Exponential(col1,error_coeffs['ColorBand1'][0],error_coeffs['ColorBand1'][1],error_coeffs['ColorBand1'][2])
        errorCol2 =  Exponential(col2,error_coeffs['ColorBand2'][0],error_coeffs['ColorBand2'][1],error_coeffs['ColorBand2'][2])
        #Noise
        noiseMag = PhotometricErrorRN[0][:incomplete_size,0] * errorMag  
        noiseCol1 = PhotometricErrorRN[0][:incomplete_size,1] * errorCol1
        noiseCol2 = PhotometricErrorRN[0][:incomplete_size,2] * errorCol2
        #Observed magnitudes
        MagObs = mag + noiseMag
        Col1Obs = col1 + noiseCol1
        Col2Obs = col2 + noiseCol2
        #Reevaluate errors
        errorMag =  Exponential(MagObs,error_coeffs['MagBand'][0],error_coeffs['MagBand'][1],error_coeffs['MagBand'][2])
        errorCol1 =  Exponential(Col1Obs,error_coeffs['ColorBand1'][0],error_coeffs['ColorBand1'][1],error_coeffs['ColorBand1'][2])
        errorCol2 =  Exponential(Col2Obs,error_coeffs['ColorBand2'][0],error_coeffs['ColorBand2'][1],error_coeffs['ColorBand2'][2])
        #Reevaluate noise
        noiseMagfilled = PhotometricErrorRN[1][:incomplete_size,0] * errorMag  
        noiseCol1filled = PhotometricErrorRN[1][:incomplete_size,1] * errorCol2
        noiseCol2filled = PhotometricErrorRN[1][:incomplete_size,2] * errorCol2
        #Observed magniudes
        Magfilled = MagObs + noiseMagfilled
        Col1filled = Col1Obs + noiseCol1filled
        Col2filled = Col2Obs + noiseCol2filled
        #Return
        return MagObs, Col1Obs, Col2Obs, Magfilled, Col1filled, Col2filled, noiseMag, noiseCol1, noiseCol2, noiseMagfilled, noiseCol1filled, noiseCol2filled
        
    def Generator(syntPop,isochrone,Nwanted,
                  binary_fraction,companion_minimum_mass_fraction,
                  d,extpar,extlaw,
                  IsoBandsDict,Obs_to_Iso_Bands,
                  photometric_limit,error_coeffs,CompInfo,
                  PopulationSamplesRN, PhotometricErrorRN):
        #
        # CREATES A SYNTHETIC CMD FROM AN ISOCHRONE
        #
        # This function serves as a HUB that calls other functions in this module for creating a synthetic CMD from an isochrone
        #   - Synthetic stars are sampled using the Kroupa (2001) mass function
        #   - A fraction of the population is formed of non-resolved binaries
        #   - Some stars are removed according to a completeness criterium
        #   - Gaussian noise is added to emulate photometric errors
        #
        # INPUTS
        #   syntPop: empty pandas DataFrame, to store the synthetic population
        #   isochrone: pandas DataFrame containing a PARSEC-COLIBRI isochrone
        #   Nwanted: size of the population (single stars + non-resolved binaries) before completeness removal
        #   binary_fraction: fraction of the synthetic population to be composed of non-resolved binaries
        #   companion_minimum_mass_fraction: minimum mass fraction for companion in binary systems
        #   d: distance (in kpc) of the population (for evaluation apparent magnitudes and errors)
        #   extpar: extinction parameter
        #   extlaw: dictionary relating the extinction parameter with the extinction coefficient and the color excess
        #   photometric_limit: fainter possible magnitude for the synthetic population
        #   error_coeffs: dictionary containing coefficients for an exponential fit of the photometric errors
        #   ObsBands: dictionary with the band names in the observational catalog
        #   IsoBandsDict: dictionary with the band names in the isochrone files
        #   Obs_to_Iso_Bands: dictionary relating the previous band names
        #   CompInfo: dictionary withFermi function parameters, describing completeness: [FermiMag,Beta,Band]
        #   PopulationSamplesRN: random numbers for sampling the population (uniformly generated between 0 and 1)
        #   PhotometricErrorRN: random numbers for sampling the population (Gaussian generated: mean = 0 ; std = 1)
        #
        #
        #Column names: isochrone
        magIso = IsoBandsDict['MagBand']
        col1Iso =IsoBandsDict['ColorBand1']
        col2Iso =IsoBandsDict['ColorBand2']
        IsoBands = [magIso,col1Iso,col2Iso]
        # Get basic information from the isochrone
        Nsolo, Nbin, mass_start, mass_end = SyntheticPopulation.IsochroneAnalyzer(isochrone,photometric_limit,d,extpar,extlaw,Nwanted,binary_fraction,IsoBands)
        #Sample single star masses
        syntPop['Mini_single'] = SyntheticPopulation.KroupaMassSampling(mass_start,mass_end,
                                                                        PopulationSamplesRN[:,0],
                                                                        Nwanted)        
        syntPop['App{}_single'.format(magIso)],syntPop['App{}_single'.format(col1Iso)],syntPop['App{}_single'.format(col2Iso)],\
            syntPop['Mass_single'] = SyntheticPopulation.InterpolateStars(isochrone,syntPop['Mini_single'],IsoBands)
        #Mark stars that will be included in non-resolved binary systems
        syntPop['IsBinary'] = [False]*Nsolo + [True]*Nbin        
        #Sample binaries
        #Mini_companion,Mag_companion,Col1_companion,Col2_companion,Mass_companion,MagTotal,Col1Total,Col2Total
        syntPop['Mini_companion'],\
        syntPop['App{}_companion'.format(magIso)],syntPop['App{}_companion'.format(col1Iso)],syntPop['App{}_companion'.format(col2Iso)],\
        syntPop['Mass_companion'],\
        syntPop['App{}'.format(magIso)],syntPop['App{}'.format(col1Iso)],syntPop['App{}'.format(col2Iso)] = \
        SyntheticPopulation.SampleBinaries(syntPop,
                                            isochrone,
                                            companion_minimum_mass_fraction,
                                            Nwanted,Nsolo,Nbin,
                                            PopulationSamplesRN[:,1],IsoBands)
        #Remove stars according to completeness
        incomplete_size = SyntheticPopulation.CompletenessRemovalFermi(syntPop,CompInfo,PopulationSamplesRN[:,2],Obs_to_Iso_Bands)
        #Add noise according to the expected photometric error
        
        syntPop[magIso],syntPop[col1Iso],syntPop[col2Iso],\
        syntPop['{}_filled'.format(magIso)],syntPop['{}_filled'.format(col1Iso)],syntPop['{}_filled'.format(col2Iso)],\
        syntPop['{}_error1'.format(magIso)],syntPop['{}_error1'.format(col1Iso)],syntPop['{}_error1'.format(col2Iso)],\
        syntPop['{}_error2'.format(magIso)],syntPop['{}_error2'.format(col1Iso)],syntPop['{}_error2'.format(col2Iso)]=\
        SyntheticPopulation.AddPhotometricUncertainty(syntPop['App{}'.format(magIso)],
                                                      syntPop['App{}'.format(col1Iso)],
                                                      syntPop['App{}'.format(col2Iso)],
                                                      error_coeffs,
                                                      PhotometricErrorRN,
                                                      incomplete_size)
        #(mag,col1,col2,error_coeffs,PhotometricErrorRN,incomplete_size)
        #Add color column
        syntPop['{}-{}'.format(col1Iso,col2Iso)] = syntPop[col1Iso] - syntPop[col2Iso]
        #dd mass column
        syntPop['Mass'] = syntPop['Mass_single'] + syntPop['Mass_companion']
        #Remove stars dimmer than the photometric limit
        syntPop.drop( syntPop[ syntPop[magIso] > photometric_limit].index,inplace=True )
        
        
        
    
class MCMCsupport:
    #Functions for helping the MCMC sampling
    
    def WalkersStartPosition(info):
        # CREATE INITIAL POSITION FOR WALKERS
        #
        # From user inputs, generate a list of initial walker positions
        #
        # INPUTS:
        #   info: dictionary with desired configurations 
        #
        # OUTPUTS:
        #   A list of lists with all possible combinations for initial parameters
        #       - Each element of the output is itself a list contaning [metalicity, age, distance, extinction parameter, binary fraction]
        #
        #Import libraries
        from numpy import linspace
        from itertools import product
        #Create lists of parameters
        mhs  = linspace( info['Metallicity']['Minimum']  , info['Metallicity']['Maximum']  , info['Metallicity']['Number'] )
        ages = linspace( info['Age']['Minimum']  , info['Age']['Maximum']  , info['Age']['Number'] )
        ds   = linspace( info['Distance']['Minimum']  , info['Distance']['Maximum']  , info['Distance']['Number'] )
        expars = linspace( info['ExtinctionPar']['Minimum']  , info['ExtinctionPar']['Maximum']  , info['ExtinctionPar']['Number'] )
        binfs = linspace( info['BinFraction']['Minimum']  , info['BinFraction']['Maximum']  , info['BinFraction']['Number'] )
        #Return list of walkers
        return list( product( mhs, ages, ds, expars, binfs ) )
        
        
    def DefinePriors(info):
        # DEFINE PRIOR FUNCTIONS
        #
        # From user inputs, define priors for each parameter
        #
        # INPUTS:
        #   info: dictionary with desired configurations 
        #
        # OUTPUTS:
        #    Prior_Metallicity,Prior_Age,Prior_Distance,Prior_ExtPar,Prior_BinFraction: functions representing the priors for each parameter
        #
        #Import libraries
        from numpy import inf,log
        #Define function for prior selection
        def SelectFunction(which):
            #Box
            if which == 'Box':
                #Define Box function
                def Box(x,params):
                    x0,x1 = params
                    #Check range
                    if x0 <= x <=x1: 
                        return 0
                    else: 
                        return -inf
                #Return function
                return Box
            elif which == 'Gaussian':
                #Define Gaussian function
                def Gauss(x,params):
                    mean,std,x0,x1 = params
                    #Check range
                    if x0 <= x <=x1: 
                        return  -0.5 * ( (x-mean)/std )**2
                    else:
                        return -inf
                #Return function
                return Gauss
            elif which == 'SuperGaussian':
            #Define Super-Gaussian function
                def SuperGauss(x,params):
                    mean,std,x0,x1 = params
                    #Check range
                    if x0 <= x <=x1: 
                        return  -0.5 * ( (x-mean)/std )**10
                    else:
                        return -inf
                #Return function
                return SuperGauss
            elif which == 'LogNormal':
                #Define Log-normal function
                def LogNorm(x,params):
                    mu,sig,x0,x1 = params
                    #Check range
                    if x0 < x <=x1: 
                        return -log(x) - (log(x)-mu)**2/(2*sig**2)
                    else: 
                        return -inf
                #Return function
                return LogNorm
        #Define priors for each variable
        Prior_Metallicity = SelectFunction(info['Metallicity']['Type'])
        Prior_Age = SelectFunction(info['Age']['Type'])
        Prior_Distance = SelectFunction(info['Distance']['Type'])
        Prior_ExtPar = SelectFunction(info['ExtinctionPar']['Type'])
        Prior_BinFraction = SelectFunction(info['BinFraction']['Type'])
        return Prior_Metallicity,Prior_Age,Prior_Distance,Prior_ExtPar,Prior_BinFraction
        
    def LikelihoodCalculator(Nclu,Nsyn ):
        # EVALUATE LIKELIHOOD
        #
        # Calculates the likelihood function defined in Tremmel+2013
        #
        # INPUTS
        #   Nclu: Hess diagram of the observed cluster
        #   Nsyn: Hess diagram of the synthetic population
        #
        # RETURNS
        #   The log-likelihood
        #
        from numpy import  log,interp
        from numpy import sum as sumall
        from scipy.special import loggamma   
        def correct(  Nobs, NobsRef, correction ):
            return interp( Nobs, NobsRef, correction )
        #Evaluate the likelihood
        return sumall( (loggamma( 0.5+Nclu+Nsyn ) - ( 0.5+Nclu+Nsyn )*log(2) - loggamma(0.5+Nsyn) - loggamma( 1+Nclu ) ) ) 