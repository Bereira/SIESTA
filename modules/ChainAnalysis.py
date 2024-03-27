#Global libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def ImportData(project_name):
    #Import libraries
    from os.path import exists
    import sys
    from numpy import loadtxt
    from emcee.backends import HDFBackend
    from pickle import load
    import shutil
    #Path to files
    path = 'projects/{}'.format(project_name)
    #Check if exists
    if exists(path):
        #Get autocorrelation time
        auto_corr_time = loadtxt(path+'/autocorrelation.dat')
        #Copy backend file
        shutil.copyfile(path+'/backend.h5', 
                        path+'/backendCopy.h5')
        #Read backend
        backend = HDFBackend(path+'/backendCopy.h5', read_only=True)
        #Open inputs
        with open(path+'/inputs.pkl','rb') as pkl_file: inputs = load(pkl_file)
    #If not, warn
    else:
        print('Project not found! Check again...')
        sys.exit()
    #Return
    return auto_corr_time, backend, inputs

def AutoCorrelation(auto_corr_time):
    #Import libraries
    from numpy import arange,diff
    from matplotlib.ticker import ScalarFormatter
    #Create figure
    fig,ax = plt.subplots(nrows=1,ncols=1,constrained_layout=True)
    #
    # FIRST PLOT
    #
    #Iteration number
    i = arange( 1, len(auto_corr_time)+1, 1  )*100
    #Plot
    ax.plot(i,auto_corr_time,c='tab:blue')
    #Labels
    ax.set_ylabel('Mean autocorrelation time',c='tab:blue')
    ax.set_xlabel('Iterations')
    #Ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #
    # SECOND PLOT
    #
    rel_diff = diff(auto_corr_time)/auto_corr_time[:-1]*100
    #Twin axis
    axt = ax.twinx()
    #Plot
    axt.plot(i[:-1],rel_diff,c='tab:red')
    #Scale
    axt.set_yscale('log')
    axt.yaxis.set_major_formatter(ScalarFormatter())
    #Label
    axt.set_ylabel('Relative difference (%)',c='tab:red',rotation=270,verticalalignment='bottom')
    #
    # SECONDARY AXIS
    #    
    def corr2uncorr(x):
        return x / auto_corr_time[-1]
    def uncorr2corr(x):
        return x * auto_corr_time[-1]
    secax = ax.secondary_xaxis('top', functions=(corr2uncorr, uncorr2corr))
    secax.set_xlabel('Uncorrelated iteractions')
    secax.xaxis.set_minor_locator(AutoMinorLocator())
    #
    # INTERACT
    #
    tau = auto_corr_time[-1]
    #Interact
    print('Final mean autocorrelation time: {:.2f}'.format(tau))
    print('\t Total iteration number is ~{:.0f} times larger than this value!'.format( 100*len(auto_corr_time) / tau))
    print('\t Final variation is {:.2f}%'.format(rel_diff[-1]))
    return tau,fig

def Samples(backend,inputs,Filter=True,walkers_show=1):
    from numpy import exp
    #Define burnin and thin
    autocorr_time = backend.get_autocorr_time(tol=0)
    burnin = int( 2*max(autocorr_time) )
    thin = int(0.5*min(autocorr_time))
    if Filter:
        #Import samples
        samples = backend.get_chain(discard=burnin, thin=thin)
        #Import likelihoods
        logprob = backend.get_log_prob(discard=burnin, thin=thin)
    else:
        #Import samples
        samples = backend.get_chain()
        #Import likelihoods
        logprob = backend.get_log_prob()
    #Number of walkers
    walker_num = len(samples[0,:,0])
    #Create figure
    fig, ax = plt.subplots(nrows=6,ncols=1,figsize=(9,8),sharex=True,constrained_layout=True)
    #Evaluate walkers step
    walkers_step = walker_num//walkers_show
    #Iterate over walkers
    for w in range(0,walker_num-(walker_num%walkers_show),walkers_step):
        #Iterate over variables
        for i in (0,1,2,3,4):
            #Plot
            ax[i].plot( samples[:,w,i],rasterized=True  )
            ax[i].plot( samples[:,:,i].mean(axis=1),rasterized=True,c='k',ls=':'  )
        ax[5].plot( logprob[:,w], rasterized=True )
        ax[5].plot( logprob[:,:].mean(axis=1),rasterized=True,c='k',ls=':'  )
    #Labels
    ax[0].set_ylabel(r'$[M/H]$')
    ax[1].set_ylabel(r'$\log Age_{yr}$')
    ax[2].set_ylabel(r'$d$ ($kpc$)')
    ax[3].set_ylabel(inputs['ExtinctionLaw']['ExtinctionParameter'])
    ax[4].set_ylabel(r'$f_{bin}$')
    ax[5].set_ylabel(r'$\ln{P}$')
    if Filter:
        ax[5].set_xlabel(r'Uncorrelated iterations')
    else:
        ax[5].set_xlabel(r'Iterations')
    #Ticks
    for i in(0,1,2,3,4,5):
        ax[i].xaxis.set_minor_locator(AutoMinorLocator())
        ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    #Return figure
    return backend.get_chain(discard=burnin, flat=True, thin=thin),logprob,fig


def MaginalPosterior(samplesIn,inputs,labels,bins,quantiles_min,quantiles_max,title,age_plot,dist_plot,obs_label,synt_label):
    #Import libraries
    from numpy import copy,linspace,quantile,argmax,median,std,pi,nan,inf,sign,absolute,exp,log10
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from astropy.visualization import hist as hist
    from scipy.optimize import curve_fit
    from scipy.stats import skewnorm
    from pickle import load    
    #Open inputs
    with open('projects/{}/inputs.pkl'.format(inputs['Project_name']),'rb') as pkl_file: inputs = load(pkl_file)
    #
    # USEFUL FUNCTIONS
    #
    def SkewNorm(x,a,loc,scale,norm):
        return norm*skewnorm.pdf(x,a, loc, scale)
    def HistSimple(ax,data,bins):
        #Plot
        counts,edges,_ = hist(data,bins=bins,histtype='step',color='k',density=True,orientation='horizontal',zorder=0)
        ax.set_xticks([])
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        #Return histogram
        return counts,edges
    def Hist2D(ax,datax,datay,binsx,binsy,filterx,filtery):
        #Filter data
        datax = datax[ filterx & filtery ]
        datay = datay[ filterx & filtery ]
        #Plot
        counts,xbins,ybins,_ = ax.hist2d(datax,datay,bins=[binsx,binsy],cmap='Greys')
        #Countour
        ax.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],cmap = plt.cm.viridis,linewidths=1,levels=4)
        #Ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    def SkewNormParams(a,loc,scale):
        #Useful parameters
        delta = a/(1+a**2)**0.5
        muz = (2/pi)**0.5*delta
        deltaz = (1-muz**2)**0.5
        skew = 0.5*(4-pi) * muz**3 / ( 1 - muz**2 )**1.5
        m0 = muz - 0.5*skew*deltaz - 0.5 * sign(a) * exp( -2*pi/absolute(a) )
        #Mode
        mode = loc + scale*m0
        #Standard deviation
        std = (scale**2*(1-muz**2))**0.5
        #Return
        return mode, std
    #
    # WORK WITH SAMPLES
    #
    #Manipulate samples, acoording to user choices
    samplesFull = copy(samplesIn)
    if age_plot == 'Gyr':
        samplesFull[:,1] = 10**(samplesFull[:,1]-9)
    if age_plot == 'Myr':
        samplesFull[:,1] = 10**(samplesFull[:,1]-6)
    if dist_plot == 'modulus':
        samplesFull[:,2] = 5*log10(samplesFull[:,2]*1000) - 5
        #Number of samples
    sampleNum = len(samplesFull[0,:])
    #Create dictionary
    samples = {}
    #Filtration index
    idx = []
    #Iterate over parameters
    for i in range(0,sampleNum):
        #Create index
        idx += [( samplesFull[:,i] >= quantile(samplesFull[:,i],quantiles_min[i]) ) & ( samplesFull[:,i] <= quantile(samplesFull[:,i],quantiles_max[i]) )]
        #Apply
        samples[i] = samplesFull[idx[i],i]
    #
    # SET LABELS AND COLUMNS
    #
    #Check if no custom labels were provided
    if labels == '':
        #Start label
        labels = [0]*sampleNum
        #Append values
        labels[0] = '[M/H]'
        if age_plot == 'Gyr': labels[1] = 'Age (Gyr)'
        elif age_plot == 'Myr': labels[1] = 'Age (Myr)'
        elif age_plot == 'log': labels[1] = r'$\log$Age$_{yr}$'
        if dist_plot == 'kpc': labels[2] = 'Dist. (kpc)'
        elif dist_plot == 'modulus': labels[2] = r'$(m-M)_0$'
        labels[3] = inputs['ExtinctionLaw']['ExtinctionParameter']
        labels[4] = 'Bin. f.'
    #Columns
    columns = [0]*sampleNum
    #Append values
    columns[0] = '[M/H]'
    if age_plot == 'Gyr': columns[1] = 'Age_Gyr'
    elif age_plot == 'Myr': columns[1] = 'Age_Myr'
    elif age_plot == 'log': columns[1] = 'logAge'
    if dist_plot == 'kpc': columns[2] = 'Dist_kpc'
    elif dist_plot == 'modulus': columns[2] = '(m-M)0'
    columns[3] = inputs['ExtinctionLaw']['ExtinctionParameter']
    columns[4] = 'BinF'
    #
    # START FIGURE
    #
    #Create figure
    fig = plt.figure(figsize=(9,7))
    #Create grid
    gs = GridSpec(5, 5, figure=fig)
    #Empty list for axes
    ax = [[None for _ in range(5)] for _ in range(6)]
    #
    # 1D MARGINALIZED POSTERIORS
    #
    #Fit parameters
    a = [0]*sampleNum
    loc = [0]*sampleNum
    scale = [0]*sampleNum
    #Final values and uncertainties
    fit_value = [0]*sampleNum
    fit_error = [0]*sampleNum
    #Image limit
    lim = []
    #1D plots
    for i in range(0,sampleNum):
        #Create subplot
        ax[i][i] = fig.add_subplot(gs[i,i])
        #Plot
        counts,edges = HistSimple(ax[i][i],samples[i],bins[i])
        #Uptdate bins
        bins[i] = len(counts)
        #Define x
        x = [(edges[i]+edges[i+1])/2 for i in range(0,len(edges)-1)]
        xx = linspace(min(edges),max(edges),1000)
        #Get limits
        lim += [[min(edges),max(edges)]]
        #Attempt to fit a skewed-normal distribution
        try:
            #Fit SKN distribution
            popt,_ = curve_fit( SkewNorm, x, counts,
                                    p0 = [0,x[argmax(counts)],samples[i].std(),1],
                                    bounds = ( [-inf,-inf,-inf,0], [inf,inf,inf,1] ) )
            a[i],loc[i],scale[i],_ = popt
            fit_value[i], fit_error[i] = SkewNormParams(a[i],loc[i],scale[i])
            #Evaluate
            skn = skewnorm(a[i], loc[i], scale[i])
            pdf = skn.pdf(xx)
            ax[i][i].plot( pdf /max(pdf)*max(counts),xx,ls='-',c='tab:blue',zorder=2)
        except:
            #Raise warning if not possible
            print('Warning! Failed to fit at index {}'.format(i))
            a[i] = nan ; loc[i] = nan ; scale[i] = nan
            #Use median instead
            fit_value[i] = median(samples[i])
            fit_error[i] = std(samples[i])
            ax[i][i].axvline(fit_value[i],zorder=2,c='tab:blue',ls='--')
            ax[i][i].axvspan(fit_value[i]-0.5*fit_error[i],fit_value[i]+0.5*fit_error[i],zorder=2,color='tab:blue',alpha=0.5,ls='--')
        #Details
        ax[i][i].yaxis.tick_right()
        ax[i][i].yaxis.set_label_position("right")
        ax[i][i].xaxis.tick_top()
        ax[i][i].xaxis.set_label_position("top")
        ax[i][i].set_xlim(ax[i][i].get_xlim()[::-1])
        ax[i][i].set_ylim(lim[i][0],lim[i][1])
        ax[i][i].tick_params(axis="both",which='both',direction="in")
    ax[sampleNum-1][sampleNum-1].set_ylabel(labels[-1])
    #
    # 2D MARGINALIZED POSTERIORS
    #
    #Iterate over ponels
    for j in range(0,sampleNum):
        for i in range(0,j):
            #Create subplot
            ax[i][j] =  fig.add_subplot(gs[i,j])
            #Plot
            Hist2D(ax[i][j],samplesFull[:,j],samplesFull[:,i],
                   bins[j],bins[i],idx[j],idx[i])
            #Labels
            if j == 4: ax[i][j].set_ylabel(labels[i])
            else: ax[i][j].tick_params(labelleft=False)
            if i == 0: ax[i][j].set_xlabel(labels[j])
            else: ax[i][j].tick_params(labelbottom=False)
            #Side
            ax[i][j].yaxis.tick_right()
            ax[i][j].yaxis.set_label_position("right")
            ax[i][j].xaxis.tick_top()
            ax[i][j].xaxis.set_tick_params(rotation=45)
            for tick in ax[i][j].xaxis.get_majorticklabels():
                tick.set_horizontalalignment("left")
            ax[i][j].xaxis.set_label_position("top")
            #Ticks
            ax[i][j].tick_params(axis="both",which='both',direction="in")
            #Axis
            ax[i][j].set_ylim( lim[i][0], lim[i][1] ) ; ax[i][j].set_xlim( lim[j][0], lim[j][1] )
    #
    # ADD TEXT
    #
    axtext = fig.add_subplot(gs[2,0:2])
    #Remove frame
    axtext.axis('off')
    #Limits
    axtext.set_xlim(0,1000)
    axtext.set_ylim(0,1000)
    #Add text
    plot_text = ''
    for i in range(0,sampleNum):
        plot_text += r'{}$ = {:.2f}\pm{:.2f}$'.format(labels[i],fit_value[i],fit_error[i])
        if i < sampleNum-1: plot_text += '\n'
    axtext.text(0.0,0.5,plot_text,
               horizontalalignment='left',verticalalignment='center',
               transform=axtext.transAxes
               )
    #
    # ADD TITLE
    #
    axtext = fig.add_subplot(gs[1,0])
    #Remove frame
    axtext.axis('off')
    #Limits
    axtext.set_xlim(0,1000)
    axtext.set_ylim(0,1000)
    axtext.text(0.5,0.5,title, weight='bold',fontsize='large',backgroundcolor='gainsboro',
               horizontalalignment='center',verticalalignment='center',
               transform=axtext.transAxes
               )  
    #
    # ADD CMDs
    #
    import MCMCsampling
    from pandas import DataFrame,read_csv,concat
    gsCMD = GridSpecFromSubplotSpec(1, 2, subplot_spec = gs[3:,0:3])
    #Catalog columns
    magObs = inputs['ObsCatalogColumns']['MagBand']
    colObs =inputs['ObsCatalogColumns']['Color']
    #Isochrone columns
    magIso = inputs['IsochroneColumns']['MagBand']
    col1Iso =inputs['IsochroneColumns']['ColorBand1']
    col2Iso =inputs['IsochroneColumns']['ColorBand2']
    colIso = '{}-{}'.format(col1Iso,col2Iso)
    #Subplots
    axCMD1 = fig.add_subplot(gsCMD[0])
    axCMD2 = fig.add_subplot(gsCMD[1])
    #Ticks
    axCMD2.tick_params(labelleft=False)
    axCMD1.tick_params(axis="both",direction="in")
    axCMD2.tick_params(axis="both",direction="in")
    #Get cluster CMD
    clusterCMD = read_csv('projects/{}/FilteredCMD.dat'.format(inputs['Project_name']), sep=',')
    #Get isochrone index
    isochroneIndex,mh_list,age_list = MCMCsampling.Initialization.GetIsochroneIndex(inputs['Grid_path'])
    #Get parameters
    #   Metallicity
    mh = round(fit_value[0],2)    
    if mh < min(mh_list):
        mh = min(mh_list)
        print('Fitted metallicity is smaller than the lower limit of the grid!')
    if mh > max(mh_list):
        mh = min(mh_list)
        print('Fitted metallicity is larger than the upper limit of the grid!')        
    #   logAge
    if age_plot == 'log':
        logAge = round(fit_value[1],2)
    elif age_plot == 'Gyr':
        logAge = round( log10(fit_value[1])+9 ,2 )
    elif age_plot == 'Myr':
        logAge = round( log10(fit_value[1])+6 ,2 )
    if logAge < min(age_list):
        logAge = min(age_list)
        print('Fitted age is smaller than the lower limit of the grid')
    if logAge > max(age_list):
        logAge = max(age_list)
        print('Fitted age is larger than the upper limit of the grid!')
    #   Distance
    if dist_plot == 'modulus':
        d =  10**( (fit_value[2] + 5)/5 )/1000
    elif dist_plot=='kpc':
        d = fit_value[2]
    if d < 0:
        d = 0
        print('Fitted distance is smaller than 0!')
    #   Extinction parameter
    extpar = max([fit_value[3],0])
    if extpar < 0:
        extpar = 0
        print('Fitted extinction parameter is smaller than 0!')
    #   Binary fraction
    if fit_value[4] < 0:
        binf = 0
        print('Fitted binary fraction is smaller than 0!')
    elif fit_value[4] > 1:
        binf = 1
        print('Fitted binary fraction is larger than 0!')
    else:
        binf = fit_value[4]
    #Import isochrone
    iso = read_csv('{}/{}.dat'.format(inputs['Grid_path'],isochroneIndex[mh][logAge]),sep=',')
    #Displace isochrone
    mu = 5*log10( d*1000 ) - 5 + extpar*inputs['ExtinctionLaw']['MagCorrection']
    CE = extpar*(inputs['ExtinctionLaw']['ColorCorrection1']-inputs['ExtinctionLaw']['ColorCorrection2'])
    iso['AppMag'] = iso[magIso] + mu
    iso['AppColor'] = iso[col1Iso]-iso[col2Iso] + CE
    #Filter isochrone
    isomini = iso[iso['AppMag']<=inputs['Photometric_limit']]
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
                                                 inputs['Completeness_Fermi'],
                                                 PopulationSamplesRN, PhotometricErrorRN)
    syntCMD = syntCMD.sample(len(clusterCMD),random_state=inputs['Seed'])
    #FIRST PLOT
    axCMD1.scatter(clusterCMD[colObs],clusterCMD[magObs],c='tab:orange',marker='.',label='Cluster',rasterized=True)
    axCMD1.scatter(syntCMD[colIso],syntCMD[magIso],c='tab:green',marker='.',label='Synt. pop.',rasterized=True)
    axCMD1.plot(isomini['AppColor'],isomini['AppMag'],c='k',ls='--')
    axCMD1.set_ylabel(r'${}$'.format(magObs))
    axCMD1.invert_yaxis()
    #axCMD1.legend()
    #SECOND PLOT
    axCMD2.scatter(syntCMD[colIso],syntCMD[magIso],c='tab:green',marker='.',label='Synt. pop.',rasterized=True)
    axCMD2.scatter(clusterCMD[colObs],clusterCMD[magObs],c='tab:orange',marker='.',label='Cluster',rasterized=True)
    axCMD2.plot(isomini['AppColor'],isomini['AppMag'],c='k',ls='--')
    #axCMD2.legend()
    axCMD2.get_shared_y_axes().join(axCMD1, axCMD2) ; axCMD2.autoscale()
    #SHARED DETAILS
    for ax in (axCMD1,axCMD2):
        ax.set_xlabel(r'${}$'.format(colObs))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis="both",which='both',direction="in")
    #
    # LABELS
    #
    ax =  fig.add_subplot(gs[4,3])
    #Ticks
    ax.xaxis.set_ticks([],[])
    ax.yaxis.set_ticks([],[])
    #Frame
    ax.axis('off')
    #Scatter
    ax.scatter([],[],c='tab:orange',marker='o',label=obs_label)
    ax.scatter([],[],c='tab:green',marker='o',label=synt_label)
    ax.legend(loc='center left')
    #
    #SAVE RESULTS
    #
    #Fitter parameters
    results = DataFrame(columns=['name','a','loc','scale','value','error'])
    results = concat( [results, DataFrame({'name':columns,
                                             'a':a,
                                             'loc':loc,
                                             'scale':scale,
                                             'value':fit_value,
                                             'error':fit_error})] )
    results.to_csv('projects/{}/results.dat'.format(inputs['Project_name']),sep='\t')
    print('Fitted parameters were stored in {}'.format('projects/{}/results.dat'.format(inputs['Project_name'])))
    #Image
    fig.savefig('projects/{}/corner.pdf'.format(inputs['Project_name']),format='pdf',dpi=600)
    print('Corner plot saved in {}'.format('projects/{}/corner.pdf'.format(inputs['Project_name']))) 
    #Return
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    return  fig


def SaveAll(inputs,figures):
     #Import libraries
    from matplotlib.backends.backend_pdf import PdfPages
    #Path name
    path = 'projects/{}'.format(inputs['Project_name'])
    
    #Image pdf
    pdf = PdfPages('{}/results.pdf'.format(path))
    #Save figures
    for fig in figures:
        pdf.savefig(fig)
    print('Check {} for the images created in this Notebook'.format(('{}/results.pdf'.format(path))))
    pdf.close()
        