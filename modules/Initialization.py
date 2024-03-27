#Global libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def ImportData(inputs):
    #Import libraries
    from os import makedirs
    from os.path import exists
    from pickle import load
    from astropy.io import fits
    from astropy.table import Table
    from pandas import DataFrame
    from numpy import isfinite,sqrt
    #Check if CLUSTER path exists
    if exists(inputs['Cluster_path']):
        #Open fits file
        hdul = fits.open(inputs['Cluster_path'])
        #Store data from file as DataFrame
        cluster_data = Table(hdul[1].data).to_pandas()
        # Close file
        hdul.close()
    #If not, interact
    else: print('WARNING! Cluster path not found!')
    #Check if GRID path exists
    if exists(inputs['Grid_path']):
        with open('{}/index.pkl'.format(inputs['Grid_path']),'rb') as pkl_file:
            #Read grid
            index = load(pkl_file)
        #List of metallicities and ages
        mh_list = sorted(list(index.keys()))
        age_list = sorted(list(index[mh_list[0]].keys()))
        #Append everython
        isochroneIndex = [index,mh_list,age_list]
    #If not, interact
    else:
        print('WARNING! Grid path not found!')
    #Warn if there is an inconsistency between the project name and the cluster path
    if not inputs['Project_name'] in inputs['Cluster_path']:
        print('WARNING! project name diverges from the indicated cluster path, check before continuing')
    #Create directory in projects/, if one does not already exists
    if not exists( 'projects/{}'.format(inputs['Project_name']) ):
        makedirs( 'projects/{}'.format(inputs['Project_name']) )
    #Warn user if a project alterady exists
    else:
        print('WARNING! There is already a project with this name. Check before continuing')
    #Get columns
    mag = inputs['ObsCatalogColumns']['MagBand']
    emag = inputs['ObsCatalogColumns']['errorMagBand']
    col1 = inputs['ObsCatalogColumns']['ColorBand1']
    ecol1 = inputs['ObsCatalogColumns']['errorColorBand1']
    col2 = inputs['ObsCatalogColumns']['ColorBand2']
    ecol2 = inputs['ObsCatalogColumns']['errorColorBand2']
    memb_prob = inputs['ObsCatalogColumns']['MembershipProbability']
    memb_mask = inputs['ObsCatalogColumns']['MembershipMask']
    #Create empty dataFrame, for storing the CMD
    clusterCMD = DataFrame()
    #Filter imported data: discard points with unavailable magnitudes and with zero membership probability
    idx = isfinite(cluster_data[mag]) & isfinite(cluster_data[col1]) & isfinite(cluster_data[col2]) & (cluster_data[memb_mask]!=0)
    #Add magnitudes and errors to the CMD dataframe
    clusterCMD[mag] = cluster_data[mag][idx]
    clusterCMD[emag] = cluster_data[emag][idx]
    clusterCMD[col1] = cluster_data[col1][idx]
    clusterCMD[ecol1] = cluster_data[ecol1][idx]
    clusterCMD[col2] = cluster_data[col2][idx]
    clusterCMD[ecol2] = cluster_data[ecol2][idx]
    #Add columns for the color
    col = '{}-{}'.format(col1,col2)
    inputs['ObsCatalogColumns']['Color'] = col
    clusterCMD[col] = clusterCMD[col1] - clusterCMD[col2]
        #Add column for the membership
    clusterCMD[memb_prob] = cluster_data[memb_prob][idx]
    #Return cluster data
    return clusterCMD, isochroneIndex

def CheckCMD(clusterCMD,inputs):
    #Import libraries
    from sklearn.neighbors import NearestNeighbors
    #
    # DEFINE COLUMN NAMES
    #
    mag = inputs['ObsCatalogColumns']['MagBand']
    col = inputs['ObsCatalogColumns']['Color']
    memb = inputs['ObsCatalogColumns']['MembershipProbability']
    #
    # REMOVE OUTLIERS FROM THE CLUSTER
    #
    #Positions renormalized by the standard deviation
    CMDrenorm = clusterCMD[[mag,col]]/clusterCMD[[mag,col]].std() 
    #Fit nearest neighbor estimator on the cluster data 
    clusterKNNestimator =  NearestNeighbors( n_neighbors=inputs['Kth-neighbor']+1 , algorithm='auto', metric='euclidean' ).fit( CMDrenorm )
    #Evaluate distance from the data itself
    distances,_  = clusterKNNestimator.kneighbors(CMDrenorm)
    #Add to cluster CMD
    clusterCMD['neighbor_dist'] = distances[:,-1]
    #Find outliers by the largest distances
    outliers = clusterCMD.nlargest(inputs['Outlier_num'],'neighbor_dist')
    #Create filtered CMD
    clusterCMD_no_outliers = clusterCMD.drop(outliers.index)
    #Remove low membership stars
    clusterCMDfilter = clusterCMD_no_outliers[clusterCMD_no_outliers[memb]>=inputs['Minimum_membership']]
    #Important numbers
    Ntotal = len(clusterCMD)
    Nselected = sum( clusterCMDfilter[mag]<=inputs['Photometric_limit'] )
    #Append to inputs
    inputs['Cluster_size'] = Nselected
    #
    # FIRST PLOT
    #
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(9,4),sharex=True,sharey=True,constrained_layout=True)
    #Custom colormap
    cmap = plt.get_cmap('binary')
    cmap.set_over('tab:pink')
    #Plot
    sc = ax[0].scatter(clusterCMD[col],clusterCMD[mag],c=clusterCMD['neighbor_dist'],cmap=cmap,
                       vmin = 0,
                       vmax = clusterCMDfilter['neighbor_dist'].max(),rasterized=True)
    #Mark next outlier
    idx = clusterCMDfilter['neighbor_dist'].argmax()
    ax[0].scatter(clusterCMDfilter[col].iloc[idx],clusterCMDfilter[mag].iloc[idx],marker='x',c='tab:pink')
    #Photometric limit
    ax[0].axhline(inputs['Photometric_limit'],ls='--',c='black')
    #Invert y axis
    ax[0].invert_yaxis()
    #Colorbar
    fig.colorbar(sc,ax = ax[0],extend='max',label='Distance to nearest star',orientation='horizontal')
    #Labels
    ax[0].set_xlabel(r'${}$'.format(col))
    ax[0].set_ylabel(r'${}$'.format(mag))
    #Ticks
    ax[0].xaxis.set_minor_locator(AutoMinorLocator())
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    #Title
    ax[0].set_title('Outlier removal')
    #
    # SECOND PLOT
    #
    #Custom colormap
    cmap = plt.get_cmap('viridis')
    cmap.set_under('tab:red')
    #Scatter
    sc = ax[1].scatter(clusterCMD_no_outliers[col],clusterCMD_no_outliers[mag],c=clusterCMD_no_outliers[memb],
                       vmin=max([inputs['Minimum_membership'],clusterCMD_no_outliers[memb].min()]),rasterized=True )
    #Photometric limit
    ax[1].axhline(inputs['Photometric_limit'],ls='--',c='black')
    #Invert y axis
    ax[1].invert_yaxis()
    #Colorbar
    fig.colorbar(sc,ax=ax[1],label='Membership',extend='min',orientation='horizontal')
    #Labels
    ax[1].set_xlabel(r'${}$'.format(col))
    ax[1].set_ylabel(r'${}$'.format(mag))
    #Ticks
    ax[1].xaxis.set_minor_locator(AutoMinorLocator())
    ax[1].yaxis.set_minor_locator(AutoMinorLocator())
    #Title
    ax[1].set_title('Cluster members')
    #
    # THIRD PLOT
    #
    #Title
    fig.suptitle('{} out of {} stars selected ({} removed)'.format( Nselected, Ntotal, Ntotal-Nselected ))
    #Scatter
    sc = ax[2].scatter(clusterCMDfilter[col],clusterCMDfilter[mag],c=clusterCMDfilter[memb],rasterized=True)
    #Photometric limit
    ax[2].axhline(inputs['Photometric_limit'],ls='--',c='black')
    #Invert y axis
    ax[2].invert_yaxis()
    #Colorbar
    fig.colorbar(sc,ax=ax[2],label='Membership',orientation='horizontal')
    #Labels
    ax[2].set_xlabel(r'${}$'.format(col))
    ax[2].set_ylabel(r'${}$'.format(mag))
    #Ticks
    ax[2].xaxis.set_minor_locator(AutoMinorLocator())
    ax[2].yaxis.set_minor_locator(AutoMinorLocator())
    #Title
    ax[2].set_title('Filtered population')
    #Ticks
    for i in (0,1,2):
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())

    return fig, clusterCMDfilter.reset_index(drop=True)

        
def ClusterBinning(clusterCMDraw,inputs):
    #Import libraries
    from scipy.special import loggamma
    from numpy import linspace,histogram2d,argmax,array,unique,arange,log
    from numpy import sum as totalsum
    #
    # DEFINE COLUMN NAMES
    #
    mag = inputs['ObsCatalogColumns']['MagBand']
    col = inputs['ObsCatalogColumns']['Color']
    memb = inputs['ObsCatalogColumns']['MembershipProbability']
    #Remove stars dimmer than the photometric limit
    clusterCMD = clusterCMDraw[ clusterCMDraw[ mag ] <= inputs['Photometric_limit']  ]
    #
    # OPTIMAL CMD BINNING
    #
    if (inputs['Color_width'] == '') or (inputs['Mag_width'] == ''): 
        # Empty list for storing results
        log_probs = []
        color_widths = []
        mag_widths = []
        #Population size
        N = len(clusterCMD)
        #Starting and final color and magnitude
        color0 = clusterCMD[col].min()
        color1 = clusterCMD[col].max()
        mag0 = clusterCMD[mag].min()
        mag1 =  inputs['Photometric_limit']
        #Start iterations
        for Ncolor in range(   2, 30 +2 ):
            for Nmag in range( 2, 30   +2):
                #Define edges
                color_edges = linspace( color0,color1, Ncolor)
                mag_edges   = linspace( mag0, mag1,   Nmag )
                #Create 2D histogram
                hist2d,_,_ = histogram2d(clusterCMD[col],clusterCMD[mag],bins=[color_edges,mag_edges],density=False)
                #Total number of bins
                M = (Ncolor-1)*(Nmag-1)
                #Calculate Knuth's posterior
                log_probs += [ N*log(M) + loggamma(M/2) - M*loggamma(0.5) - loggamma(N+M/2) + totalsum(loggamma(hist2d+0.5)) ]
                #Append witdhs
                color_widths += [ color_edges[1] - color_edges[0] ]
                mag_widths   += [ mag_edges[1]   - mag_edges[0] ]
        #Best widths
        color_widthBest = color_widths[argmax(log_probs)]
        mag_widthBest = mag_widths[argmax(log_probs)]
        inputs['Color_width'] = color_widthBest 
        inputs['Mag_width'] = mag_widthBest
        print('OPTIMAL BINNING')
        print('\t Color: {:.4f}'.format(color_widthBest))
        print('\t Magnitude: {:.4f}'.format(mag_widthBest))
    #Edges for final histogram
    inputs['Edges_color'] = arange(clusterCMD[col].min() -  inputs['Color_width'],
                                   clusterCMD[col].max() + 2* inputs['Color_width'],
                                    inputs['Color_width'])
    inputs['Edges_magnitude'] = arange(inputs['Photometric_limit'],
                                       clusterCMD[mag].min() - 2* inputs['Mag_width'],
                                       - inputs['Mag_width'])[::-1]
    #
    # FIGURE
    #
    #
    # FIRST PLOT 
    #
    #Create figure
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,5),constrained_layout=True)
    #Plot CMD 
    sc = ax.scatter(clusterCMD[col],clusterCMD[mag],c=clusterCMD[memb],rasterized=True)
    #Photometric limit
    ax.axhline(inputs['Photometric_limit'],c='k',ls='--')
    #Colorbar
    fig.colorbar(sc,ax=ax,orientation='horizontal',label='Membership probability')
    #Labels
    ax.set_xlabel(r'${}$'.format(col))
    ax.set_ylabel(r'${}$'.format(mag))
    #Limits
    ax.set_xlim(inputs['Edges_color'][0],inputs['Edges_color'][-1])
    ax.set_ylim(inputs['Edges_magnitude'][-1],inputs['Edges_magnitude'][0])
    #Ticks
    ax.set_xticks(inputs['Edges_color'],minor=True) 
    ax.set_yticks(inputs['Edges_magnitude'],minor=True)
    ax.tick_params(axis='both',which='minor',colors='None')
    #Grid
    ax.grid(which='minor',c='tab:grey',ls=':')
    ax.xaxis.remove_overlapping_locs = False
    ax.yaxis.remove_overlapping_locs = False
    #Return
    return fig

def ClusterDensity(clusterCMD,inputs):
    #Import libraries
    from numpy import concatenate,zeros
    from pandas import DataFrame
    from scipy.stats import truncnorm
    import MCMCsampling
    #
    # DEFINE COLUMN NAMES
    #
    mag = inputs['ObsCatalogColumns']['MagBand']
    emag = inputs['ObsCatalogColumns']['errorMagBand']
    col1 = inputs['ObsCatalogColumns']['ColorBand1']
    ecol1 = inputs['ObsCatalogColumns']['errorColorBand1']
    col2 = inputs['ObsCatalogColumns']['ColorBand2']
    ecol2 = inputs['ObsCatalogColumns']['errorColorBand2']
    memb_prob = inputs['ObsCatalogColumns']['MembershipProbability']
    col =inputs['ObsCatalogColumns']['Color']
    #FUNCTION TO GENERATE GRID
    def GridGenerator(clusterCMD):
        #Filter photometric limit
        clusterCMDbrighter = clusterCMD[ clusterCMD[mag] <= phot_lim ]
        #Create CMD grid
        CMDgrid = MCMCsampling.Distribution.Evaluate(colors, magnitudes, clusterCMDbrighter[col], clusterCMDbrighter[mag],
                                                       renorm_factor=inputs['Cluster_size']/len(clusterCMDbrighter))
        #Return CMDgrid
        return CMDgrid
    #Get the filling factor
    filling_factor = inputs['Filling_factor']
    #Get the photometric limit
    phot_lim = inputs['Photometric_limit']
    #Get edges
    colors = inputs['Edges_color']
    magnitudes = inputs['Edges_magnitude']
    #
    # FILL DATA AND RENORMALIZE IT
    #
    #Define truncated normal distribution
    def TN(mean,std,low,up):
        return truncnorm( (low-mean)/std, (up-mean)/std, loc=mean, scale=std)
    #Dictionary with filled data
    filled_data = {}
    #Add magnitudes
    filled_data[mag] = concatenate( [ clusterCMD[mag].iloc[i] + \
                                      concatenate([[0],clusterCMD[emag].iloc[i]*TN(mean=0,std=1,low=-3,up=3).rvs(filling_factor-1,
                                                                                                               random_state=1)])\
                                      for i in clusterCMD.index])
    filled_data[col1] = concatenate( [ clusterCMD[col1].iloc[i] + \
                                      concatenate([[0],clusterCMD[ecol1].iloc[i]*TN(mean=0,std=1,low=-3,up=3).rvs(filling_factor-1,
                                                                                                               random_state=2)])\
                                      for i in clusterCMD.index])
    filled_data[col2] = concatenate( [ clusterCMD[col2].iloc[i] + \
                                      concatenate([[0],clusterCMD[ecol2].iloc[i]*TN(mean=0,std=1,low=-3,up=3).rvs(filling_factor-1,
                                                                                                               random_state=3)])\
                                     for i in clusterCMD.index])
    #Add memberships
    filled_data[memb_prob] = concatenate( [ clusterCMD[memb_prob].iloc[i] + zeros(filling_factor) for i in clusterCMD.index])
    #Evaluate colors
    filled_data[col] = filled_data[col1] - filled_data[col2]
    #Create data frame
    clusterCMDfilled = DataFrame( filled_data )
    #
    # CREATE THE CMD GRID
    #
    #Generate
    CMDgrid = GridGenerator(clusterCMD)
    CMDgridFilled = GridGenerator(clusterCMDfilled)
    inputs['Cluster_size_eff'] = CMDgridFilled.sum()
    #
    # FIRST PLOT
    #
    #Create figure
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(9,4),constrained_layout=True)
    #Plot CMD 
    sc = ax[0].scatter(clusterCMD[col],clusterCMD[mag],c=clusterCMD[memb_prob],rasterized=True)
    #Photometric limit
    ax[0].axhline(phot_lim,c='k',ls='--')
    ax[1].axhline(phot_lim,c='k',ls='--')
    ax[2].axhline(phot_lim,c='k',ls='--')
    #Colorbar
    fig.colorbar(sc,ax=ax[0],orientation='horizontal',label='Membership probability')
    #Labels
    ax[0].set_xlabel(r'${}$'.format(col))
    ax[0].set_ylabel(r'${}$'.format(mag))
    #Limits
    ax[0].set_xlim(colors[0],colors[-1])
    ax[0].set_ylim(magnitudes[-1],magnitudes[0])
    #Ticks
    ax[0].set_xticks(colors,minor=True) 
    ax[0].set_yticks(magnitudes,minor=True)
    ax[0].tick_params(axis='both',which='minor',colors='None')
    #Grid
    ax[0].grid(which='minor',c='tab:grey',ls=':')
    ax[0].xaxis.remove_overlapping_locs = False
    ax[0].yaxis.remove_overlapping_locs = False
    #Title
    ax[0].set_title('Filtered CMD')
    #
    # SECOND PLOT
    #
    #Color limits
    cmin = min([ CMDgrid.min(), CMDgridFilled.min() ])
    cmax = max([ CMDgrid.max(), CMDgridFilled.max() ])
    
    #Density map
    hist1 = ax[1].imshow( CMDgrid, cmap='Blues', vmin = cmin, vmax= cmax,
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
    #Colorbar
    fig.colorbar(hist1,ax=ax[1],orientation='horizontal',label='Counts')
    #Labels
    ax[1].set_xlabel(r'${}$'.format(col))
    ax[1].set_ylabel(r'${}$'.format(mag))
    #Title
    ax[1].set_title('CMD distribution')
    #
    # THIRD PLOT
    #
    #Density map
    hist2 = ax[2].imshow( CMDgridFilled ,  cmap='Blues', vmin = cmin, vmax= cmax,
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
    #Colorbar
    fig.colorbar(hist2,ax=ax[2],orientation='horizontal',label='Counts')
    #Labels
    ax[2].set_xlabel(r'${}$'.format(col))
    ax[2].set_ylabel(r'${}$'.format(mag))
    #Title
    ax[2].set_title('CMD distribution (filled)')
    
    #Return
    return fig, clusterCMD, clusterCMDfilled ,CMDgridFilled

def ImportIsochrone(clusterCMD,inputs,mh,logage,d,extpar,isochroneIndex):
    #Import libraries
    from pandas import read_csv
    from numpy import log10
    #
    # DEFINE COLUMN NAMES
    #
    #Catalog
    magObs = inputs['ObsCatalogColumns']['MagBand']
    col1Obs = inputs['ObsCatalogColumns']['ColorBand1']
    col2Obs = inputs['ObsCatalogColumns']['ColorBand2']
    colObs =inputs['ObsCatalogColumns']['Color']
    memb_prob = inputs['ObsCatalogColumns']['MembershipProbability']
    #Isochrone
    magIso = inputs['IsochroneColumns']['MagBand']
    col1Iso =inputs['IsochroneColumns']['ColorBand1']
    col2Iso =inputs['IsochroneColumns']['ColorBand2']
    #
    # Relate band in observed catalog and in the isochrones
    #
    inputs['Bands_Obs_to_Iso'] = {  magObs : magIso,
                                    col1Obs: col1Iso,
                                    col2Obs: col2Iso}
    #
    # IMPORT ISOCHRONE
    #
    #Read file
    iso = read_csv('{}/{}.dat'.format(inputs['Grid_path'],isochroneIndex[mh][logage]),sep=',')
    #Evaluate distance modulus and color excess
    mu0 = 5*log10( d*1000 ) - 5 
    Amag =  extpar*inputs['ExtinctionLaw']['MagCorrection']
    Acol1 =  extpar*inputs['ExtinctionLaw']['ColorCorrection1']
    Acol2 =  extpar*inputs['ExtinctionLaw']['ColorCorrection2']
    CE = extpar*(Acol1-Acol2)
    #Print
    print('Extinction parameter ({}): {}'.format(inputs['ExtinctionLaw']['ExtinctionParameter'],extpar))
    print('\t Extinction coefficient in the {} band: {}'.format(magIso,Amag ))
    print('\t Extinction coefficient in the {} band: {}'.format(col1Iso,Acol1 ))
    print('\t Extinction coefficient in the {} band: {}'.format(col2Iso,Acol2 ))
    print('\t Color excess coefficient in {}-{}: {}'.format(col1Iso,col2Iso,CE ))
    #Dispalce isochrone
    iso['AppMag'] = iso[magIso] + mu0 + Amag
    iso['AppColor'] = iso[col1Iso]-iso[col2Iso] + CE
    #Filter isochrone
    isomini = iso[iso['AppMag']<=inputs['Photometric_limit']]
    #
    # PLOT
    #
    #Create figure
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4), constrained_layout=True )
    #
    # FIRST PLOT
    #
    #Scatter
    sc = ax.scatter(clusterCMD[colObs],clusterCMD[magObs],c=clusterCMD[memb_prob],rasterized=True)
    #Colorbar
    fig.colorbar(sc,ax=ax,orientation='vertical',label='Membership probability')
    #Plot
    ax.plot(isomini['AppColor'],isomini['AppMag'],c='tab:red')
    #Details
    ax.set_ylim(top=inputs['Photometric_limit'])
    ax.invert_yaxis()
    ax.set_xlabel(r'${}$'.format(colObs))
    ax.set_ylabel(r'${}$'.format(magObs))
    #Return figure
    return fig

def FermiCompleteness(clusterCMD,inputs):
    #Import libraries
    from astropy.visualization import hist
    from numpy import exp,linspace,array
    #
    # DEFINE COLUMN NAMES
    #
    mag = inputs['ObsCatalogColumns']['MagBand']
    #Band
    band = inputs['Completeness_Fermi']['Band']
    #Define Fermi Function
    def FermiFunction(x,xf,beta):
        return 1 / (1+exp( beta*(x-xf)  ))
    #Filter photometric limit
    clusterCMDbrighter = clusterCMD[ clusterCMD[mag] <= inputs['Photometric_limit'] ]
    #Create figure
    fig,ax = plt.subplots(nrows=1,ncols=1,constrained_layout=True)
    #Histogram
    h = hist(clusterCMDbrighter[band],bins='knuth',color='tab:blue',label='Observed number',align='mid',histtype='stepfilled')
    #Bin center and width
    bins = array( [ 0.5*(h[1][i]+h[1][i+1]) for i in range(0,len(h[1])-1) ] )
    width = h[1][1] - h[1][0]
    #Compensate completeness
    corrected_num = h[0] /  FermiFunction( bins, inputs['Completeness_Fermi']['FermiMag'],inputs['Completeness_Fermi']['Beta'])
    #Plot
    ax.bar( bins, corrected_num,width=width,edgecolor='tab:red',color='None',zorder=-1,align='center' ,label=r'Observerd number $\times$ Completeness$^{-1}$'  )
    #Fermi function
    axt = ax.twinx()
    magnitudes = linspace( clusterCMD[band].min(),clusterCMD[band].max(),1000 )
    axt.plot( magnitudes,FermiFunction(magnitudes,inputs['Completeness_Fermi']['FermiMag'],inputs['Completeness_Fermi']['Beta']),
             c='tab:red')
    #Interact
    Nmembers = sum(h[0])
    Ncomplete = sum(corrected_num)
    print('Number of members: {}'.format(Nmembers))
    print('Completeness corrected number: ~{:.0f}'.format(Ncomplete))
    print('\t(every star counts as 1/completeness)')
    print('Expected loss: ~{:.2f}%'.format( (Ncomplete-Nmembers)/Nmembers*100 ))
    #Legend
    ax.legend(loc='upper right')
    #Invert axis
    ax.invert_xaxis()
    #Labels
    ax.tick_params(axis='y', labelcolor='tab:blue')
    axt.tick_params(axis='y', labelcolor='tab:red')
    ax.set_ylabel('Number of cluster stars',color='tab:blue')
    axt.set_ylabel('Completeness',color='tab:red')
    ax.set_xlabel(r'${}$'.format(band))
    #Ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    axt.yaxis.set_minor_locator(AutoMinorLocator())
    #Return figure
    return fig

def ErrorCurve(clusterCMD,inputs):
    #Import specific libraries
    from scipy.optimize import curve_fit
    #Exponential function
    def Exponential(x,a,b,c):
        from numpy import exp
        return a * exp( b * x ) + c
    # Bands
    bands = ['MagBand','ColorBand1','ColorBand2']
    #Dictionary with results
    error_coeffs = {}
    #Create figure 
    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(9,3),constrained_layout=True)
    #Iterate over bands
    for i in range(0,len(bands)):
        #Magnitude
        mag =  inputs['ObsCatalogColumns'][bands[i]]
        emag = inputs['ObsCatalogColumns']['error'+bands[i]]
        #Select data
        data = clusterCMD[[mag,emag]].copy().sort_values(mag).dropna().reset_index(drop=True)
        #Apply rolling quantile
        mag_quantile = data[mag].rolling(inputs['{}_points_rolling_mean'.format(bands[i])]).mean().dropna()
        emag_quantile = data[emag].rolling(inputs['{}_points_rolling_mean'.format(bands[i])]).quantile(inputs['{}_quantile'.format(bands[i])]).dropna()
        #Scatter plot
        ax[i].scatter(clusterCMD[mag],clusterCMD[emag],marker='$\u25EF$',c='k',label='Data',zorder=-1,rasterized=True)
        ax[i].scatter(mag_quantile,emag_quantile,marker='+',c='tab:red',label='Moving mean',zorder=1,rasterized=True)
        #Labels
        ax[i].set_xlabel(r'Measured ${}$'.format(mag))
        ax[i].set_ylabel(r'Error in ${}$'.format(mag))
        ax[i].set_title(bands[i])
        #Fit error function
        try:
            #Fit
            coeff,_ = curve_fit(Exponential,mag_quantile, emag_quantile)
            #Append to dictionary
            error_coeffs[bands[i]] = coeff
            #Plot fit
            ax[i].plot(data[mag],Exponential(data[mag],coeff[0],coeff[1],coeff[2]),c='tab:cyan',label='Custom fit',zorder=2,ls='-')
        #Raise warning if fit was not possible
        except:
            print('Warning! Unable to find a fit for {} band! Must solve this issue before proceeding'.format(bands[i]))
        #Append to inputs
        inputs['Error_coefficients'] = error_coeffs
    #Return figure
    return fig
   

    
def SyntheticPopulation(clusterCMD,ClusterGrid,inputs,isochroneIndex,pop_params):
    #Import libraries
    import MCMCsampling
    from numpy import exp,inf,log10
    from pandas import DataFrame
    from pandas import read_csv
    #
    # DEFINE COLUMN NAMES
    #
    #Catalog
    magObs = inputs['ObsCatalogColumns']['MagBand']
    colObs =inputs['ObsCatalogColumns']['Color']
    #Isochrone
    magIso = inputs['IsochroneColumns']['MagBand']
    col1Iso =inputs['IsochroneColumns']['ColorBand1']
    col2Iso =inputs['IsochroneColumns']['ColorBand2']
    colIso = '{}-{}'.format(col1Iso,col2Iso)
    inputs['IsochroneColumns']['Color'] = colIso
    #
    # CREATE SYNTHETIC POPULATION
    #
    #Get population parameters
    mh,age,d,extpar,binf = pop_params
    #Isochrone
    iso = read_csv('{}/{}.dat'.format(inputs['Grid_path'],isochroneIndex[mh][age]),sep=',')
    #Random numbers for synthetic population
    PopulationSamplesRN, PhotometricErrorRN = MCMCsampling.Initialization.RandomNumberStarter(inputs['Initial_population_size'],
                                                                                               inputs['ObsCatalogColumns'])
    #Empty Dataframe for synthetic population
    syntCMD = DataFrame()
    #Generate synthetic population
    MCMCsampling.SyntheticPopulation.Generator(syntCMD,
                                                 iso.copy(), 
                                                 inputs['Initial_population_size'],
                                                 binf, inputs['Companion_min_mass_fraction'],
                                                 d,extpar,inputs['ExtinctionLaw'],
                                                 inputs['IsochroneColumns'],inputs['Bands_Obs_to_Iso'],
                                                 inputs['Photometric_limit'],
                                                 inputs['Error_coefficients'], 
                                                 {'FermiMag': inputs['Photometric_limit']+1,'Beta':inf,'Band':magObs },
                                                 PopulationSamplesRN, PhotometricErrorRN)
    #
    # INCOMPLETE POPULATION
    #
    # Define Fermi function
    def FermiFunction(x,xf,beta):
        return 1 / (1+exp( beta*(x-xf)  ))
    #Evaluate completeness for synthetic population
    syntCMD['Completeness'] = FermiFunction(syntCMD['App{}'.format( inputs['Bands_Obs_to_Iso'][inputs['Completeness_Fermi']['Band']])],
                                            inputs['Completeness_Fermi']['FermiMag'],
                                            inputs['Completeness_Fermi']['Beta'])
    #Drop stars randomly, proportionaly to the completeness
    syntCMD['Keep'] =  PopulationSamplesRN[:len(syntCMD),2] <= syntCMD['Completeness']
    #Final population
    finalCMD = syntCMD[syntCMD['Keep']].copy()
    #Dispalce isochrone
    mu = 5*log10( d*1000 ) - 5 + extpar*inputs['ExtinctionLaw']['MagCorrection']
    CE = extpar*(inputs['ExtinctionLaw']['ColorCorrection1']-inputs['ExtinctionLaw']['ColorCorrection2'])
    iso['AppMag'] = iso[magIso] + mu
    iso['AppColor'] = iso[col1Iso]-iso[col2Iso] + CE
    #Filter isochrone
    isomini = iso[iso['AppMag']<=inputs['Photometric_limit']]
    #
    #FIGURE
    #
    #Create figure
    fig = plt.figure(constrained_layout=True,figsize=(9,6))
    #Create subfigures
    FIGtemp,FIGprocess = fig.subfigures(nrows=1,ncols=2)
    #Sample figures
    FIGsample,FIGhist = FIGtemp.subfigures(nrows=2,ncols=1)
    #Titles
    FIGprocess.suptitle('Synthetic population generation')
    FIGsample.suptitle('Sample comparisson')
    FIGhist.suptitle('Histogram comparisson')
    #
    # PLOT SAMPLES
    #
    ax = FIGsample.subplots(nrows=1,ncols=2,sharex=True,sharey=True)
    #Sample population
    sampleCMD = finalCMD.sample(n=int(inputs['Cluster_size']),random_state=inputs['Seed'])
    #Scatter
    
    ax[0].scatter(sampleCMD[colIso],sampleCMD[magIso],facecolor='None',edgecolor='tab:red',label='Synt.',rasterized=True)
    ax[0].scatter(clusterCMD[colObs],clusterCMD[magObs],facecolor='None',edgecolor='tab:blue',label='Obs.',rasterized=True)
    ax[0].legend()
    ax[1].scatter(clusterCMD[colObs],clusterCMD[magObs],facecolor='None',edgecolor='tab:blue',rasterized=True)
    ax[1].scatter(sampleCMD[colIso],sampleCMD[magIso],facecolor='None',edgecolor='tab:red',rasterized=True)
    #Iterate over plots
    for i in (0,1):
        #Isochrone
        ax[i].plot(isomini['AppColor'],isomini['AppMag'],c='k',ls=':')
        #Label
        ax[i].set_xlabel(r'${}$'.format(colObs))
        ax[i].set_ylabel(r'${}$'.format(magObs))
        #Ticks
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    #Invert axis
    ax[0].invert_yaxis()
    #
    # PLOT PROCESS
    #
    ax = FIGprocess.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
    #Iterate over plots
    for i in (0,1):
        for j in (0,1):
            #Isochrone
            ax[i,j].plot(isomini['AppColor'],isomini['AppMag'],c='k',ls=':')
            #Label
            ax[i,j].set_xlabel(r'${}$'.format(colObs))
            ax[i,j].set_ylabel(r'${}$'.format(magObs))
            #Ticks
            ax[i,j].xaxis.set_minor_locator(AutoMinorLocator())
            ax[i,j].yaxis.set_minor_locator(AutoMinorLocator())
        #Invert axis
    ax[0,0].invert_yaxis()
    #IMF SAMPLING
    ax[0,0].set_title('IMF')
    ax[0,0].scatter(syntCMD['App{}_single'.format(col1Iso)]-syntCMD['App{}_single'.format(col2Iso)],
                    syntCMD['App{}_single'.format(magIso)],marker='.',s=50,c='tab:purple',
                   rasterized=True)
    ax[0,0].plot([],[],c='k',ls=':',label='Isochrone')
    ax[0,0].scatter([],[],marker='o',c='tab:purple',label='Single stars')
    ax[0,0].legend(loc='upper left')
    #BINARIES
    ax[0,1].set_title('Binaries')
    bins = syntCMD[syntCMD['IsBinary']]
    solo = syntCMD[~ syntCMD['IsBinary']]
    ax[0,1].scatter(solo['App{}_single'.format(col1Iso)]-solo['App{}_single'.format(col2Iso)],solo['App{}_single'.format(magIso)],marker='.',s=50,c='tab:purple',rasterized=True)
    ax[0,1].scatter(bins['App{}'.format(col1Iso)]-bins['App{}'.format(col2Iso)],bins['App{}'.format(magIso)],marker='.',s=50,c='tab:green',rasterized=True)
    ax[0,1].scatter([],[],marker='o',c='tab:green',label='Binaries')
    ax[0,1].legend()
    #COMPLETENESS
    ax[1,0].set_title('Completeness')
    bins = syntCMD[(syntCMD['Keep']) & (syntCMD['IsBinary'])]
    solo = syntCMD[(syntCMD['Keep']) & ~(syntCMD['IsBinary'])]
    rejected = syntCMD[~(syntCMD['Keep'])]
    ax[1,0].scatter(solo['App{}_single'.format(col1Iso)]-solo['App{}_single'.format(col2Iso)],solo['App{}_single'.format(magIso)],marker='.',s=50,c='tab:purple',rasterized=True)
    ax[1,0].scatter(bins['App{}'.format(col1Iso)]-bins['App{}'.format(col2Iso)],bins['App{}'.format(magIso)],marker='.',s=50,c='tab:green',rasterized=True)
    ax[1,0].scatter(rejected['App{}'.format(col1Iso)]-rejected['App{}'.format(col2Iso)],rejected['App{}'.format(magIso)],marker='.',s=50,c='tab:orange',rasterized=True)
    ax[1,0].scatter([],[],marker='o',c='tab:orange',label='Removed')
    ax[1,0].legend()
    #ERRORS
    ax[1,1].set_title('Photometric error')
    ax[1,1].scatter(solo[colIso],solo[magIso],marker='.',s=50,c='tab:purple',rasterized=True)
    ax[1,1].scatter(bins[colIso],bins[magIso],marker='.',s=50,c='tab:green',rasterized=True)
    #
    # PLOT HISTOGRAMS
    #
    ax = FIGhist.subplots(nrows=1,ncols=3,sharex=True,sharey=True)
    #Get edges
    colors = inputs['Edges_color']
    magnitudes = inputs['Edges_magnitude']
    #Synt distribution
    SyntGrid = MCMCsampling.Distribution.Evaluate(colors, magnitudes, 
                                                  finalCMD['{}_filled'.format(col1Iso)]-finalCMD['{}_filled'.format(col2Iso)],
                                                  finalCMD['{}_filled'.format(magIso)],
                                                  renorm_factor=inputs['Cluster_size']/len(finalCMD))
    
    #Color limits
    cmin = min( [ SyntGrid.min(),ClusterGrid.min()  ] )
    cmax = max( [ SyntGrid.max(),ClusterGrid.max()  ] )
    #Density map
    hist1 = ax[0].imshow( ClusterGrid, cmap='Blues', vmin = cmin, vmax= cmax,
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
    hist2 = ax[1].imshow( SyntGrid, cmap='Reds', vmin = cmin, vmax= cmax,
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
    Delta = (ClusterGrid-SyntGrid)
    cref = max( -Delta.min(),Delta.max() )
    hist3 = ax[2].imshow( Delta, cmap='coolwarm_r', vmin = -cref, vmax= cref,
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
    #Titles
    ax[0].set_title('Observed')
    ax[1].set_title('Synthetic')
    ax[2].set_title(r'Obs.$-$Synt.')
    
    #Colorbar
    fig.colorbar(hist1,ax=ax[0],orientation='horizontal',label='Counts')
    fig.colorbar(hist2,ax=ax[1],orientation='horizontal',label='Counts')
    fig.colorbar(hist3,ax=ax[2],orientation='horizontal',label=r'Counts')
    #Iterate over plots
    for i in (0,1,2):
        #Label
        ax[i].set_xlabel(r'${}$'.format(colObs))
        ax[i].set_ylabel(r'${}$'.format(magObs))
    #Return
    return fig
    
    
def Priors(inputs,mh_list,age_list):
    #Import libraries
    from numpy import linspace,inf,exp,log
    #Define functions
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
            return Box,params
        elif which == 'Gaussian':
            #Define Gaussian function
            def Gauss(x,params):
                mean,std,x0,x1 = params
                #Check range
                if x0 <= x <=x1: 
                    return -0.5 * ( (x-mean)/std )**2
                else: 
                    return -inf
            #Return function
            return Gauss,params[2:]
        elif which == 'LogNormal':
            def LogNorm(x,params):
                mu,sig,x0,x1 = params
                #Check range
                if x0 < x <=x1: 
                    return -log(x) - (log(x)-mu)**2/(2*sig**2) +log(0.14)+(log(0.14)-mu)**2/(2*sig**2)
                else: 
                    return -inf
            #Return function
            return LogNorm,params[2:]
    #Create figure
    fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(9,2  ),constrained_layout=True,sharey=True)
    #
    # METALLICITY
    #
    #Get function type and parameters
    which,params =  inputs['Priors']['Metallicity']['Type'] , inputs['Priors']['Metallicity']['Parameters']
    #Select function
    f,xlim = SelectFunction(which)
    #x values
    x = linspace(xlim[0],xlim[1],10000)
    #Evaluate
    y = [f(x,params) for x in x]
    #Figure
    ax[0].plot(x,exp( y ))
    for i in 0,1: ax[0].axvline(xlim[i],c='k',ls='--')
    #Title
    ax[0].set_title('Metallicity')
    #Label
    ax[0].set_xlabel(r'$[M/H]$')
    ax[0].set_ylabel(r'Probability ')
    ax[0].set_yticklabels('')
    #Ticks
    for i in(0,1,2,3):
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    #
    # AGE
    #
    #Get function type and parameters
    which,params =  inputs['Priors']['Age']['Type'] , inputs['Priors']['Age']['Parameters']
    #Select function
    f,xlim = SelectFunction(which)
    #x values
    x = linspace(xlim[0],xlim[1],1000)
    #Evaluate
    y = [f(x,params) for x in x]
    #Figure
    ax[1].plot(x,exp( y ))
    for i in 0,1: ax[1].axvline(xlim[i],c='k',ls='--')
    #Title
    ax[1].set_title('Age')
    #Label
    ax[1].set_xlabel(r'$\log Age_{Gyr}$')
    #
    # DISTANCE
    #
    #Get function type and parameters
    which,params =  inputs['Priors']['Distance']['Type'] , inputs['Priors']['Distance']['Parameters']
    #Select function
    f,xlim = SelectFunction(which)
    #x values
    x = linspace(xlim[0],xlim[1],1000)
    #Evaluate
    y = [f(x,params) for x in x]
    #Figure
    ax[2].plot(x,exp( y ))
    for i in 0,1: ax[2].axvline(xlim[i],c='k',ls='--')
    #Title
    ax[2].set_title('Distance')
    #Label
    ax[2].set_xlabel(r'$d$ (kpc)')
    #
    # Extinction parameter
    #
    #Get function type and parameters
    which,params =  inputs['Priors']['ExtinctionPar']['Type'] , inputs['Priors']['ExtinctionPar']['Parameters']
    #Select function
    f,xlim = SelectFunction(which)
    #x values
    x = linspace(xlim[0],xlim[1],1000)
    #Evaluate
    y = [f(x,params) for x in x]
    #Figure
    ax[3].plot(x,exp( y ))
    for i in 0,1: ax[3].axvline(xlim[i],c='k',ls='--')
    #Title
    ax[3].set_title(inputs['ExtinctionLaw']['ExtinctionParameter'])
    #Label
    ax[3].set_xlabel(inputs['ExtinctionLaw']['ExtinctionParameter'])
    #
    # BINARY FRACTION
    #
    #Get function type and parameters
    which,params =  inputs['Priors']['BinFraction']['Type'] , inputs['Priors']['BinFraction']['Parameters']
    #Select function
    f,xlim = SelectFunction(which)
    #x values
    x = linspace(xlim[0],xlim[1],1000)
    #Evaluate
    y = [f(x,params) for x in x]
    #Figure
    ax[4].plot(x,exp( y ))
    for i in 0,1: ax[4].axvline(xlim[i],c='k',ls='--')
    #Title
    ax[4].set_title('Binary fraction')
    #Label
    ax[4].set_xlabel(r'Fraction')
    #Return figure
    return fig
    
    
def StartingWalkers(inputs,mh_list,age_list):
    from numpy import linspace,meshgrid,inf,exp,array,log
    from matplotlib.gridspec import GridSpec
    #Number of walkers
    walker_num = inputs['Walkers_start']['Metallicity']['Number']*\
                 inputs['Walkers_start']['Age']['Number']*\
                 inputs['Walkers_start']['Distance']['Number']*\
                 inputs['Walkers_start']['ExtinctionPar']['Number']*\
                 inputs['Walkers_start']['BinFraction']['Number']
    print('Total number of walkers: {}'.format( walker_num ) )
    print('Maximum size of the chain: {}'.format(walker_num*inputs['Max_iterations']))     
    #
    # PLOT FUNCTIONS
    #
    #Create figure
    fig = plt.figure(constrained_layout=True,figsize=(9,7))
    #Create grid
    gs = GridSpec(5, 5, figure=fig)
    #Empty array
    ax = [[None] * 5 for _ in range(5)]  
    def PlotSolo(ax,data):
        #Plot
        ax.scatter(data,[0.5]*len(data),c='tab:blue',marker='x',zorder=1)
        #Ticks
        ax.set_yticks([])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        #Limits
        ax.set_ylim(-0.1,1.1)
        
    def PlotPair(ax,data1,data2):
        #Create pairs
        y,x = meshgrid(data1,data2)
        #Plot
        ax.scatter(x,y,c='tab:blue',marker='x')
        #Ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    #Quantities list 
    quantities = ['Metallicity', 'Age', 'Distance','ExtinctionPar','BinFraction']
    datas = [ linspace(inputs['Walkers_start']['{}'.format(q)]['Minimum'], \
                       inputs['Walkers_start']['{}'.format(q)]['Maximum'],\
                       inputs['Walkers_start']['{}'.format(q)]['Number'] ) for q in quantities ]
    labels = [r'$[M/H]$', r'$\log Age_{yr}$', r'$d$ ($kpc$)', inputs['ExtinctionLaw']['ExtinctionParameter'],'Bin. fraction']
    #
    # SOLO PLOTS
    #
    for i in (0,1,2,3,4):
        #Create subplot
        ax[i][i] = fig.add_subplot(gs[i,i])
        #Plot figure
        PlotSolo(ax[i][i],datas[i])
        #Label
        if i == 4: ax[i][i].set_xlabel(labels[i])
    #
    # DOUBLE PLOTS
    #
    for i in (1,2,3,4):
        for j in range(0,i):
            #Create subplot
            ax[i][j] = fig.add_subplot(gs[i,j])
            #Plot figure
            PlotPair(ax[i][j],datas[i],datas[j])
            #labels
            if i == 4: ax[i][j].set_xlabel(labels[j])
            if j == 0: ax[i][j].set_ylabel(labels[i])
    #
    # PLOT PRIORS
    #
    #Define functions
    def SelectFunction(which,params):
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
            return Box,params
        elif which == 'Gaussian':
            #Define Gaussian function
            def Gauss(x,params):
                mean,std,x0,x1 = params
                #Check range
                if x0 <= x <=x1: 
                    return -0.5 * ( (x-mean)/std )**2
                else: 
                    return -inf
            #Return function
            return Gauss,params[2:]
        elif which == 'LogNormal':
            def LogNorm(x,params):
                mu,sig,x0,x1 = params
                #Check range
                if x0 < x <=x1: 
                    return -log(x) - (log(x)-mu)**2/(2*sig**2) +log(0.14)+(log(0.14)-mu)**2/(2*sig**2)
                else: 
                    return -inf
            #Return function
            return LogNorm,params[2:]
    #Get function type and parameters
    #
    which,mh_params =  inputs['Priors']['Metallicity']['Type'] , inputs['Priors']['Metallicity']['Parameters']
    mh_f,mh_lims = SelectFunction(which,mh_params)
    mh = linspace(mh_lims[0],mh_lims[1],1000)
    mh_p = array( [exp(mh_f(x,mh_params)) for x in mh] )
    #
    which,age_params =  inputs['Priors']['Age']['Type'] , inputs['Priors']['Age']['Parameters']
    age_f,age_lims = SelectFunction(which,age_params)
    age = linspace(age_lims[0],age_lims[1],1000)
    age_p = array([exp(age_f(x,age_params)) for x in age])
    #
    which,d_params =  inputs['Priors']['Distance']['Type'] , inputs['Priors']['Distance']['Parameters']
    d_f,d_lims = SelectFunction(which,d_params)
    d = linspace(d_lims[0],d_lims[1],1000)
    d_p = array([exp(d_f(x,d_params)) for x in d])
    #
    which,expar_params =  inputs['Priors']['ExtinctionPar']['Type'] , inputs['Priors']['ExtinctionPar']['Parameters']
    expar_f,expar_lims = SelectFunction(which,expar_params)
    expar = linspace(expar_lims[0],expar_lims[1],1000)
    expar_p = array([exp(expar_f(x,expar_params)) for x in expar])
    #
    which,binf_params =  inputs['Priors']['BinFraction']['Type'] , inputs['Priors']['BinFraction']['Parameters']
    binf_f,binf_f_lims = SelectFunction(which,binf_params)
    binf = linspace(binf_f_lims[0],binf_f_lims[1],1000)
    binf_p = array([exp(binf_f(x,binf_params)) for x in binf])
    #1D Plots
    ax[0][0].plot(mh,mh_p,c='tab:gray',zorder=-1)
    ax[1][1].plot(age,age_p,c='tab:gray',zorder=-1)
    ax[2][2].plot(d,d_p,c='tab:gray',zorder=-1)
    ax[3][3].plot(expar,expar_p,c='tab:gray',zorder=-1)
    ax[4][4].plot(binf,binf_p,c='tab:gray',zorder=-1)
    #2D plots
    from numpy import outer
    mh_age = outer( mh_p,age_p) ; mh_ageX,mh_ageY = meshgrid(mh,age)
    mh_d = outer(mh_p,d_p) ;  mh_dX,mh_dY = meshgrid(mh,d)
    mh_expar = outer(mh_p,expar_p) ; mh_exparX,mh_exparY = meshgrid(mh,expar)
    mh_binf = outer(mh_p,binf_p) ; mh_binfX,mh_binfY = meshgrid(mh,binf)
    age_d = outer(age_p,d_p) ; age_dX, age_dY =  meshgrid(age,d)
    age_expar = outer(age_p,expar_p) ; age_exparX, age_exparY = meshgrid(age,expar)
    age_binf = outer(age_p,binf_p) ; age_binfX, age_binfY = meshgrid(age,binf)
    d_expar = outer(d_p,expar_p) ; d_exparX, d_exparY = meshgrid(d,expar)
    d_binf = outer(d_p,binf_p) ; d_binfX, d_binfY = meshgrid(d,binf)
    expar_binf = outer(expar_p,binf_p) ; expar_binfX, expar_binfY = meshgrid(expar,binf)
    ax[1][0].contourf(mh_ageX,mh_ageY,mh_age.T,zorder=-1,cmap='Greys',vmin=0,vmax=1)
    ax[2][0].contourf(mh_dX,mh_dY,mh_d.T,zorder=-1,cmap='Greys',vmin=0,vmax=1)
    ax[3][0].contourf(mh_exparX,mh_exparY,mh_expar.T,zorder=-1,cmap='Greys',vmin=0,vmax=1)
    ax[4][0].contourf(mh_binfX,mh_binfY,mh_binf.T,zorder=-1,cmap='Greys',vmin=0,vmax=1)
    ax[2][1].contourf(age_dX,age_dY,age_d.T,zorder=-1,cmap='Greys',vmin=0,vmax=1)
    ax[3][1].contourf(age_exparX,age_exparY,age_expar.T,zorder=-1,cmap='Greys',vmin=0,vmax=1)
    ax[4][1].contourf(age_binfX,age_binfY,age_binf.T,zorder=-1,cmap='Greys',vmin=0,vmax=1)
    ax[3][2].contourf(d_exparX,d_exparY,d_expar.T,zorder=-1,cmap='Greys',vmin=0,vmax=1)
    ax[4][2].contourf(d_binfX,d_binfY,d_binf.T,zorder=-1,cmap='Greys',vmin=0,vmax=1)
    ax[4][3].contourf(expar_binfX,expar_binfY,expar_binf.T,zorder=-1,cmap='Greys',vmin=0,vmax=1)
    #Return
    return fig    
    
        

def SaveAll(inputs,figures,clusterCMD,ClusterGrid):
    #Import libraries
    from numpy import savetxt
    from os.path import exists
    from os import mkdir
    from pickle import dump
    from matplotlib.backends.backend_pdf import PdfPages
    #Path name
    path = 'projects/{}'.format(inputs['Project_name'])
    #Check if path exists and, if not, create it
    if not exists(path): mkdir(path)
    #
    # INPUTS
    #
    #Save inputs
    filename = 'projects/{}/inputs.dat'.format(inputs['Project_name'])
    inputsfile = open(filename,'w')
    for k in inputs.keys():
        inputsfile.write('{}:'.format(k))
        inputsfile.write('\t{}\n'.format(inputs[k]))
    inputsfile.close()
    with open('{}/inputs.pkl'.format(path), 'wb') as file: dump(inputs, file)
    #
    # IMAGES
    #
    #Image pdf
    pdf = PdfPages('{}/inputs.pdf'.format(path))
    #Save figures
    for fig in figures:
        pdf.savefig(fig)
    pdf.close()
    #
    # FILES
    #
    #Save CMD
    filename = 'projects/{}/FilteredCMD.dat'.format(inputs['Project_name'])
    clusterCMD.to_csv(filename)
    #Save CMD grid
    filename = 'projects/{}/ClusterGrid.dat'.format(inputs['Project_name'])
    savetxt(filename,ClusterGrid)
    #Interact
    print('Inputs saved! Check files in projects/{}'.format(inputs['Project_name']))
    print('\t inputs.dat and inputs.pkl contain the choosen inputs')
    print('\t inputs.pdf contains the corresponding figures')
    print('\t FilteredCMD.dat contains the filtered cluster CMD')
    print('\t CMDgrid.dat contains the grid with the cluster distribution')
