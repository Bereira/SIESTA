#Global libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def ImportData(project_name):
    #Import libraries
    from os.path import exists
    from h5py import File
    from os import remove
    from numpy import loadtxt
    from emcee.backends import HDFBackend
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
        #Read
        backend = HDFBackend(path+'/backendCopy.h5', read_only=True)
        #If not, warn
    else:
        print('Project not found! Check again...')
    #Return
    return auto_corr_time, backend

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

def Samples(backend,Filter,walkers_show):
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
    ax[3].set_ylabel(r'$E(B-V)$')
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
    return samples,logprob,fig


def Corner(backend,project,age_plot,distance_plot,seed,title='',synt_label='Synt. pop.',obs_label='Cluster'):
    from numpy import median,quantile,argmax,round, log10, column_stack, linspace,exp
    from numpy import histogram as histogram
    from pandas import DataFrame,concat
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Rectangle
    from astropy.visualization import hist as hist
    from scipy.stats import skewnorm
    from scipy.optimize import curve_fit
    from numpy import pi,sign,absolute,exp,inf
    from pickle import load
    import SIESTAmodules
    #
    # USEFUL FUNCTIONS
    #
    def SkewNorm(x,a,loc,scale,norm):
        return norm*skewnorm.pdf(x,a, loc, scale)
    def HistSimple(ax,data):
        #Plot
        h = hist(data,bins=30,histtype='step',color='k',orientation='horizontal')
        ax.set_xticks([])
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        #Return histogram
        return h
    def Hist2D(ax,datax,datay,best_value,median_value):
        #Plot
        counts,xbins,ybins,image = ax.hist2d(datax,datay,bins=30,cmap='Greys')
        #Countour
        ax.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],cmap = plt.cm.viridis,linewidths=1,levels=4)
        #Scatter
        #ax.scatter(best_value[0],best_value[1],c='tab:green',marker='x')
        #ax.scatter(median_value[0],median_value[1],c='tab:red',marker='+')
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
    # IMPORT DATA
    #
    #Define burnin and thin
    autocorr_time = backend.get_autocorr_time(tol=0)
    burnin = int( 2*max(autocorr_time) )
    thin = int(0.5*min(autocorr_time))
    #Import samples
    samples = backend.get_chain(discard=burnin, flat=True, thin=thin)
    samples_full = backend.get_chain(flat=True)
    #Convert logAge to age
    ages = 10**(samples[:,1]-9) ; samples = column_stack((samples,ages))
    ages_full = 10**(samples_full[:,1]-9) ; samples_full = column_stack((samples_full,ages_full))
    #Convert d to distance modules
    mu0 = 5*log10( samples[:,2]*1000 ) - 5 ; samples = column_stack((samples,mu0))
    mu0_full = 5*log10( samples_full[:,2]*1000 ) - 5 ; samples_full = column_stack((samples_full,mu0_full))
    #Import likelihoods
    logprob_full = backend.get_log_prob(flat=True)
    #
    # FIND SOLUTION
    #
    # Medians
    median_values = median(samples,axis=0)
    upper_uncertainty = quantile( samples, 0.84, axis=0 ) - median_values
    lower_uncertainty = median_values - quantile( samples, 0.16, axis=0 )
    #Empty lists for kew norm params
    fit_values  = [None for _ in range(7)] 
    fit_uncertainty = [None for _ in range(7)] 
    skew_norm_params = [None for _ in range(7)] 
    #Empty dataframe to store results
    results = DataFrame(columns=['name','a','loc','scale','value','error'])
    names = ['MH','logAge','d','red','binf','age','mu']
    for i in (0,1,2,3,4,5,6):
        h,xedges = histogram(samples[:,i],bins=30,density=True)
        x = [(xedges[i]+xedges[i+1])/2 for i in range(0,len(xedges)-1)]
        try:
            popt,pcov = curve_fit( SkewNorm, x, h,
                                   p0 = [0,x[argmax(h)],samples[:,i].std(),1],
                                   bounds = ( [-inf,-inf,-inf,0], [inf,inf,inf,1] ) )
            a,loc,scale,norm = popt
        except:
            print('Warning! Failed to fit at index {}'.format(i))
            a,loc,scale = skewnorm.fit(samples[:,i],
                                       loc= x[argmax(h)],
                                       scale=samples[:,i].std())
        fit_values[i], fit_uncertainty[i] = SkewNormParams(a,loc,scale)
        skew_norm_params[i] = [a,loc,scale]
        

        
        #Save values
        results = concat( [results, DataFrame({'name':[names[i]],
                                             'a':[a],
                                             'loc':[loc],
                                             'scale':[scale],
                                             'value':[fit_values[i]],
                                             'error':[fit_uncertainty[i]]})] )
    results.set_index('name',inplace=True)
    results.to_csv('projects/{}/results.dat'.format(project),sep='\t')
    #Interact: median
    print('Median values (1 sigma quantiles for uncertainties)')
    print('\t [M/H] = {:.2f} +{:.2f} -{:.2f}'.format(median_values[0],upper_uncertainty[0],lower_uncertainty[0]))
    print('\t logAge = {:.2f} +{:.2f} -{:.2f} \t Age (Gyr) = {:.2f} +{:.2f} -{:.2f}'.format(median_values[1],upper_uncertainty[1],lower_uncertainty[1],median_values[5],upper_uncertainty[5],lower_uncertainty[5]))
    print('\t dist = {:.2f} +{:.2f} -{:.2f} \t (m-M)0 = {:.2f} +{:.2f} -{:.2f}'.format(median_values[2],upper_uncertainty[2],lower_uncertainty[2],median_values[6],upper_uncertainty[6],lower_uncertainty[6]))
    print('\t E(B-V) = {:.2f} +{:.2f} -{:.2f}'.format(median_values[3],upper_uncertainty[3],lower_uncertainty[3]))
    print('\t Bin. fraction = {:.2f} +{:.2f} -{:.2f}'.format(median_values[4],upper_uncertainty[4],lower_uncertainty[4]))
    #Best fit
    best_values = round( samples_full[argmax(logprob_full),:], 2)
    #Interact: best value
    print('Best values (max posterior):')
    print('\t [M/H] = {}'.format(best_values[0]))
    print('\t logAge = {} \t Age (Gyr) = {:.2f}'.format(best_values[1],best_values[5]))
    print('\t dist = {} \t (m-M)0 = {:.2f}'.format(best_values[2],best_values[6]))
    print('\t E(B-V) = {}'.format(best_values[3]))
    print('\t Bin. fraction = {}'.format(best_values[4]))
    #Interact: fit
    print('Skewed normal distribution fit (mode and standard deviation):')
    print('\t [M/H] = {:.2f} +/- {:.2f}'.format(fit_values[0],fit_uncertainty[0]))
    print('\t logAge = {:.2f} +/- {:.2f} \t Age (Gyr) = {:.2f} +/- {:.2f}'.format(fit_values[1],fit_uncertainty[1],fit_values[5],fit_uncertainty[5]))
    print('\t dist = {:.2f} +/- {:.2f} \t (m-M)0 =  {:.2f} +/- {:.2f}'.format(fit_values[2],fit_uncertainty[2],fit_values[6],fit_uncertainty[6]))
    print('\t E(B-V) = {:.2f} +/- {:.2f}'.format(fit_values[3],fit_uncertainty[3]))
    print('\t Bin. fraction =  {:.2f} +/- {:.2f}'.format(fit_values[4],fit_uncertainty[4]))
    #Some more info
    print('Fit properties')
    print('\t[M/H]:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[0][0],
                                                                          skew_norm_params[0][1],
                                                                          skew_norm_params[0][2]))
    print('\tlogAge:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[1][0],
                                                                          skew_norm_params[1][1],
                                                                          skew_norm_params[1][2]))
    print('\t\tAge:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[5][0],
                                                                          skew_norm_params[5][1],
                                                                          skew_norm_params[5][2]))
    print('\tdist:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[2][0],
                                                                          skew_norm_params[2][1],
                                                                          skew_norm_params[2][2]))
    print('\t(m-M)0:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[6][0],
                                                                          skew_norm_params[6][1],
                                                                          skew_norm_params[6][2]))
    print('\tE(B-V):\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[3][0],
                                                                          skew_norm_params[3][1],
                                                                          skew_norm_params[3][2]))
    print('\tBin. F.:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[4][0],
                                                                          skew_norm_params[4][1],
                                                                          skew_norm_params[4][2]))
    #
    # PRIORS
    #
    #Indexes
    indexes = ['Metallicity','Age','Distance','Reddening','BinFraction']
    #Open inputs
    with open('projects/{}/inputs.pkl'.format(project),'rb') as pkl_file: inputs = load(pkl_file)
    #Get priors
    priors = SIESTAmodules.MCMCsupport.DefinePriors(inputs['Priors'])
    #
    # PLOT HISTOGRAMS
    #
    #Label dictionary
    labels = {0:r'$[M/H]$',
              1:r'$\log Age_{yr}$',
              2:r'$d$ ($kpc$)',
              3:r'$E(B-V)$',
              4:r'Bin. fraction'}
    #Check how distance and age must be plotted
    fit_values_plot  = fit_values.copy()
    if age_plot == 'linear':
        samples[:,1] = samples[:,5]
        fit_values_plot[1] = fit_values[5]
        fit_uncertainty[1] = fit_uncertainty[5]
        skew_norm_params[1]  = skew_norm_params[5]
        labels[1] = r'$Age (Gyr)$'
    if distance_plot == 'modulus':
        samples[:,2] = samples[:,6]
        fit_values_plot[2] = fit_values[6]
        fit_uncertainty[2] = fit_uncertainty[6]
        skew_norm_params[2]  = skew_norm_params[6]
        labels[2] = r'$(m-M)_0$'
    #Create figure
    fig = plt.figure(figsize=(9,7))
    #plt.rcParams.update({'font.size': 12})

    #Create grid
    gs = GridSpec(5, 5, figure=fig)
    #Empty list for axes
    ax = [[None for _ in range(5)] for _ in range(6)]
    #Maximum value of histograms
    max_counts = []
    #Axis limits
    lim = []
    #1D plots
    for i in (0,1,2,3,4):
        #Create subplot
        ax[i][i] = fig.add_subplot(gs[i,i])
        #Plot
        h = HistSimple(ax[i][i],samples[:,i])
        #Define the function
        a,loc,scale = skew_norm_params[i]
        skn = skewnorm(a, loc, scale)
        #Get max value
        max_counts+=[max(h[0])]
        #Plot limits
        ax[i][i].set_ylim(min(h[1]),max(h[1]))
        lim += [[min(h[1]),max(h[1])]]#ax[i][i].get_ylim()
        #Get x
        y = linspace(min(h[1]),max(h[1]),1000)
        #Get y
        if (i==1) & (age_plot=='linear'):
            prior = exp([ priors[i](yy,inputs['Priors'][indexes[i]]['Parameters']) for yy in log10(y)+9 ] )
        elif (i==2) & (distance_plot=='modulus'):
            prior = exp([ priors[i](yy,inputs['Priors'][indexes[i]]['Parameters']) for yy in 10**( (y+5)/5-3 ) ] )
        else:
            prior = exp([ priors[i](yy,inputs['Priors'][indexes[i]]['Parameters']) for yy in y ] )
        pdf = skn.pdf(y)
        #Plot prior
        ax[i][i].plot(prior/max(prior)*max_counts[i],y,ls=':',c='k')
        #Plot fitted skewnorm
        ax[i][i].plot( pdf /max(pdf)*max_counts[i],y,ls='-',c='tab:blue',zorder=-1)
        #Span
        '''
        ax[i][i].errorbar( max_counts[i]/2, fit_values_plot[i],
                           yerr = fit_uncertainty[i],
                           fmt='o',c='tab:red',capsize=3,elinewidth=1,markersize=3)
        '''
        if i<4:ax[i][i].tick_params(labelleft=False)
        #Side
        ax[i][i].yaxis.tick_right()
        ax[i][i].yaxis.set_label_position("right")
        ax[i][i].xaxis.tick_top()
        ax[i][i].xaxis.set_label_position("top")
        ax[i][i].set_xlim(ax[i][i].get_xlim()[::-1])
        #Ticks
        ax[i][i].tick_params(axis="both",which='both',direction="in")
    ax[4][4].set_ylabel(labels[4])
    #2D plots
    for j in (1,2,3,4):
        for i in range(0,j):
            #Create subplot
            ax[i][j] =  fig.add_subplot(gs[i,j])
            #Plot
            Hist2D(ax[i][j],samples[:,j],samples[:,i],
                   [best_values[j],best_values[i]],
                   [median_values[j],median_values[i]])
            #Labels
            if j == 4: ax[i][j].set_ylabel(labels[i])
            else: ax[i][j].tick_params(labelleft=False)
            if i == 0: ax[i][j].set_xlabel(labels[j])
            else: ax[i][j].tick_params(labelbottom=False)
            '''
            #Scatter
            ax[i][j].errorbar( fit_values_plot[j], fit_values_plot[i],
                               xerr = fit_uncertainty[j],
                               yerr = fit_uncertainty[i],
                               fmt='o',c='tab:red',capsize=3,elinewidth=1,markersize=3)
            '''
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
            #ax[i][j].get_shared_y_axes().join(ax[i][j], ax[i][i]) ; ax[i][j].autoscale()
    #Axis details
    '''
    for i in (0,1,2,3,4):
        #Share axis
        ax[i][i].get_shared_y_axes().join(ax[i][i], ax[i][4]) ; ax[i][i].autoscale()
        '''
    #Iterate over plots       
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
    plot_text = '$[M/H] = {:.2f}\pm{:.2f}$\n'.format(fit_values_plot[0],fit_uncertainty[0]) 
    if age_plot == 'linear':
        plot_text += 'Age$= {:.2f}\pm{:.2f}$ Gyr \n'.format(fit_values_plot[1],fit_uncertainty[1])
    else: 
        plot_text += '$\log Age_{{Gyr}} = {:.2f}\pm{:.2f}$\n'.format(fit_values_plot[1],fit_uncertainty[1])
    if distance_plot == 'modulus':
        plot_text += '$(m-M)_0 = {:.2f}\pm{:.2f}$\n'.format(fit_values_plot[2],fit_uncertainty[2])
    else:
        plot_text += '$d = {:.2f}\pm{:.2f}$ kpc \n'.format(fit_values_plot[2],fit_uncertainty[2])
    plot_text += '$E(B-V)= {:.2f}\pm{:.2f}$\n'.format(fit_values_plot[3],fit_uncertainty[3]) 
    plot_text +='Bin.F.$= {:.2f}\pm{:.2f}$'.format(fit_values_plot[4],fit_uncertainty[4])
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
    # ADD CMD
    #
    import SIESTAmodules
    from pandas import DataFrame,read_csv
    gsCMD = GridSpecFromSubplotSpec(1, 2, subplot_spec = gs[3:,0:3])
    #Subplots
    axCMD1 = fig.add_subplot(gsCMD[0])
    axCMD2 = fig.add_subplot(gsCMD[1])
    #Ticks
    axCMD2.tick_params(labelleft=False)
    axCMD1.tick_params(axis="both",direction="in")
    axCMD2.tick_params(axis="both",direction="in")
    #Get cluster CMD
    clusterCMD = read_csv('projects/{}/FilteredCMD.dat'.format(project), sep=',')
    #Get isochrone index
    isochroneIndex,_,_ = SIESTAmodules.Initialization.GetIsochroneIndex(inputs['Grid_path'])
    #Isochrone
    mh = round(fit_values[0],2)
    age = round(fit_values[1],2)
    iso = read_csv('{}/{}.dat'.format(inputs['Grid_path'],isochroneIndex[mh][age]),sep=',')
    #Displace isochrone
    DM_V,DM_I,E_VI = SIESTAmodules.Auxiliary.CMDshift(fit_values[2],fit_values[3])
    iso['Vapp'] = iso['Vmag'] + DM_V
    iso['Iapp'] = iso['Imag'] + DM_I
    isomini = iso[iso['Vapp'] <= inputs['Photometric_limit']]
    #Random numbers for synthetic population
    PopulationSamplesRN, PhotometricErrorRN = SIESTAmodules.Initialization.RandomNumberStarter(inputs['Initial_population_size'],inputs['Seed'])
    #Synthetic population
    syntCMD = DataFrame()
    SIESTAmodules.SyntheticPopulation.Generator(syntCMD,
                                                 iso.copy(), 
                                                 inputs['Initial_population_size'],
                                                 fit_values[4], 
                                                 inputs['Companion_min_mass_fraction'],
                                                 fit_values[2],fit_values[3],
                                                 inputs['Photometric_limit'],
                                                 inputs['Error_coefficients'], 
                                                 inputs['Completeness_Fermi'],
                                                 PopulationSamplesRN, PhotometricErrorRN)
    syntCMD = syntCMD.sample(len(clusterCMD),random_state=seed)
    #FIRST PLOT
    axCMD1.scatter(clusterCMD['V-I'],clusterCMD['V'],c='tab:orange',marker='.',label='Cluster',rasterized=True)
    axCMD1.scatter(syntCMD['V']-syntCMD['I'],syntCMD['V'],c='tab:green',marker='.',label='Synt. pop.',rasterized=True)
    axCMD1.plot(isomini['Vapp']-isomini['Iapp'],isomini['Vapp'],c='k',ls='--')
    axCMD1.set_ylabel(r'$V$')
    axCMD1.invert_yaxis()
    #axCMD1.legend()
    #SECOND PLOT
    axCMD2.scatter(syntCMD['V']-syntCMD['I'],syntCMD['V'],c='tab:green',marker='.',label='Synt. pop.',rasterized=True)
    axCMD2.scatter(clusterCMD['V-I'],clusterCMD['V'],c='tab:orange',marker='.',label='Cluster',rasterized=True)
    axCMD2.plot(isomini['Vapp']-isomini['Iapp'],isomini['Vapp'],c='k',ls='--')
    #axCMD2.legend()
    axCMD2.get_shared_y_axes().join(axCMD1, axCMD2) ; axCMD2.autoscale()
    #SHARED DETAILS
    for ax in (axCMD1,axCMD2):
        ax.set_xlabel(r'$V-I$')
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
    #
    #
    #Return
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    return best_values, median_values, fit_values, fig,samples

def CornerPTBR(backend,project,age_plot,distance_plot,seed,title='',synt_label='Synt. pop.',obs_label='Cluster'):
    from numpy import median,quantile,argmax,round, log10, column_stack, linspace,exp
    from numpy import histogram as histogram
    from pandas import DataFrame,concat
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Rectangle
    from astropy.visualization import hist as hist
    from scipy.stats import skewnorm
    from scipy.optimize import curve_fit
    from numpy import pi,sign,absolute,exp,inf
    from pickle import load
    import SIESTAmodules
    import locale
    #
    # USEFUL FUNCTIONS
    #
    def SkewNorm(x,a,loc,scale,norm):
        return norm*skewnorm.pdf(x,a, loc, scale)
    def HistSimple(ax,data):
        #Plot
        h = hist(data,bins=30,histtype='step',color='k',orientation='horizontal')
        ax.set_xticks([])
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        #Return histogram
        return h
    def Hist2D(ax,datax,datay,best_value,median_value):
        #Plot
        counts,xbins,ybins,image = ax.hist2d(datax,datay,bins=30,cmap='Greys',rasterized=True)
        #Countour
        ax.contour(counts.transpose(),extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]],cmap = plt.cm.viridis,linewidths=1,levels=4)
        #Scatter
        #ax.scatter(best_value[0],best_value[1],c='tab:green',marker='x')
        #ax.scatter(median_value[0],median_value[1],c='tab:red',marker='+')
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
    # IMPORT DATA
    #
    #Define burnin and thin
    autocorr_time = backend.get_autocorr_time(tol=0)
    burnin = int( 2*max(autocorr_time) )
    thin = int(0.5*min(autocorr_time))
    #Import samples
    samples = backend.get_chain(discard=burnin, flat=True, thin=thin)
    samples_full = backend.get_chain(flat=True)
    #Convert logAge to age
    ages = 10**(samples[:,1]-9) ; samples = column_stack((samples,ages))
    ages_full = 10**(samples_full[:,1]-9) ; samples_full = column_stack((samples_full,ages_full))
    #Convert d to distance modules
    mu0 = 5*log10( samples[:,2]*1000 ) - 5 ; samples = column_stack((samples,mu0))
    mu0_full = 5*log10( samples_full[:,2]*1000 ) - 5 ; samples_full = column_stack((samples_full,mu0_full))
    #Import likelihoods
    logprob_full = backend.get_log_prob(flat=True)
    #
    # FIND SOLUTION
    #
    # Medians
    median_values = median(samples,axis=0)
    upper_uncertainty = quantile( samples, 0.84, axis=0 ) - median_values
    lower_uncertainty = median_values - quantile( samples, 0.16, axis=0 )
    #Empty lists for kew norm params
    fit_values  = [None for _ in range(7)] 
    fit_uncertainty = [None for _ in range(7)] 
    skew_norm_params = [None for _ in range(7)] 
    #Empty dataframe to store results
    results = DataFrame(columns=['name','a','loc','scale','value','error'])
    names = ['MH','logAge','d','red','binf','age','mu']
    for i in (0,1,2,3,4,5,6):
        h,xedges = histogram(samples[:,i],bins=30,density=True)
        x = [(xedges[i]+xedges[i+1])/2 for i in range(0,len(xedges)-1)]
        try:
            popt,pcov = curve_fit( SkewNorm, x, h,
                                   p0 = [0,x[argmax(h)],samples[:,i].std(),1],
                                   bounds = ( [-inf,-inf,-inf,0], [inf,inf,inf,1] ) )
            a,loc,scale,norm = popt
        except:
            print('Warning! Failed to fit at index {}'.format(i))
            a,loc,scale = skewnorm.fit(samples[:,i],
                                       loc= x[argmax(h)],
                                       scale=samples[:,i].std())
        fit_values[i], fit_uncertainty[i] = SkewNormParams(a,loc,scale)
        skew_norm_params[i] = [a,loc,scale]
        

        
        #Save values
        results = concat( [results, DataFrame({'name':[names[i]],
                                             'a':[a],
                                             'loc':[loc],
                                             'scale':[scale],
                                             'value':[fit_values[i]],
                                             'error':[fit_uncertainty[i]]})] )
    results.set_index('name',inplace=True)
    #Interact: median
    print('Median values (1 sigma quantiles for uncertainties)')
    print('\t [M/H] = {:.2f} +{:.2f} -{:.2f}'.format(median_values[0],upper_uncertainty[0],lower_uncertainty[0]))
    print('\t logAge = {:.2f} +{:.2f} -{:.2f} \t Age (Gyr) = {:.2f} +{:.2f} -{:.2f}'.format(median_values[1],upper_uncertainty[1],lower_uncertainty[1],median_values[5],upper_uncertainty[5],lower_uncertainty[5]))
    print('\t dist = {:.2f} +{:.2f} -{:.2f} \t (m-M)0 = {:.2f} +{:.2f} -{:.2f}'.format(median_values[2],upper_uncertainty[2],lower_uncertainty[2],median_values[6],upper_uncertainty[6],lower_uncertainty[6]))
    print('\t E(B-V) = {:.2f} +{:.2f} -{:.2f}'.format(median_values[3],upper_uncertainty[3],lower_uncertainty[3]))
    print('\t Bin. fraction = {:.2f} +{:.2f} -{:.2f}'.format(median_values[4],upper_uncertainty[4],lower_uncertainty[4]))
    #Best fit
    best_values = round( samples_full[argmax(logprob_full),:], 2)
    #Interact: best value
    print('Best values (max posterior):')
    print('\t [M/H] = {}'.format(best_values[0]))
    print('\t logAge = {} \t Age (Gyr) = {:.2f}'.format(best_values[1],best_values[5]))
    print('\t dist = {} \t (m-M)0 = {:.2f}'.format(best_values[2],best_values[6]))
    print('\t E(B-V) = {}'.format(best_values[3]))
    print('\t Bin. fraction = {}'.format(best_values[4]))
    #Interact: fit
    print('Skewed normal distribution fit (mode and standard deviation):')
    print('\t [M/H] = {:.2f} +/- {:.2f}'.format(fit_values[0],fit_uncertainty[0]))
    print('\t logAge = {:.2f} +/- {:.2f} \t Age (Gyr) = {:.2f} +/- {:.2f}'.format(fit_values[1],fit_uncertainty[1],fit_values[5],fit_uncertainty[5]))
    print('\t dist = {:.2f} +/- {:.2f} \t (m-M)0 =  {:.2f} +/- {:.2f}'.format(fit_values[2],fit_uncertainty[2],fit_values[6],fit_uncertainty[6]))
    print('\t E(B-V) = {:.2f} +/- {:.2f}'.format(fit_values[3],fit_uncertainty[3]))
    print('\t Bin. fraction =  {:.2f} +/- {:.2f}'.format(fit_values[4],fit_uncertainty[4]))
    #Some more info
    print('Fit properties')
    print('\t[M/H]:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[0][0],
                                                                          skew_norm_params[0][1],
                                                                          skew_norm_params[0][2]))
    print('\tlogAge:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[1][0],
                                                                          skew_norm_params[1][1],
                                                                          skew_norm_params[1][2]))
    print('\t\tAge:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[5][0],
                                                                          skew_norm_params[5][1],
                                                                          skew_norm_params[5][2]))
    print('\tdist:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[2][0],
                                                                          skew_norm_params[2][1],
                                                                          skew_norm_params[2][2]))
    print('\t(m-M)0:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[6][0],
                                                                          skew_norm_params[6][1],
                                                                          skew_norm_params[6][2]))
    print('\tE(B-V):\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[3][0],
                                                                          skew_norm_params[3][1],
                                                                          skew_norm_params[3][2]))
    print('\tBin. F.:\t a = {:.4g},\t loc = {:.4g},\t scale = {:.4g}'.format(skew_norm_params[4][0],
                                                                          skew_norm_params[4][1],
                                                                          skew_norm_params[4][2]))
    #
    # PRIORS
    #
    #Indexes
    indexes = ['Metallicity','Age','Distance','Reddening','BinFraction']
    #Open inputs
    with open('projects/{}/inputs.pkl'.format(project),'rb') as pkl_file: inputs = load(pkl_file)
    #Get priors
    priors = SIESTAmodules.MCMCsupport.DefinePriors(inputs['Priors'])
    #
    # PLOT HISTOGRAMS
    #
    #Label dictionary
    labels = {0:r'$[M/H]$',
              1:r'$\log Idade_{anos}$',
              2:r'$d$ ($kpc$)',
              3:r'$E(B-V)$',
              4:r'Frac. bin.'}
    #Check how distance and age must be plotted
    fit_values_plot  = fit_values.copy()
    if age_plot == 'linear':
        samples[:,1] = samples[:,5]
        fit_values_plot[1] = fit_values[5]
        fit_uncertainty[1] = fit_uncertainty[5]
        skew_norm_params[1]  = skew_norm_params[5]
        labels[1] = r'Idade (Ganos)'
    if distance_plot == 'modulus':
        samples[:,2] = samples[:,6]
        fit_values_plot[2] = fit_values[6]
        fit_uncertainty[2] = fit_uncertainty[6]
        skew_norm_params[2]  = skew_norm_params[6]
        labels[2] = r'$(m-M)_0$'
    #Create figure
    fig = plt.figure(figsize=(9,7))#,constrained_layout=True)
    locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')
    plt.rcdefaults()
    plt.rcParams['axes.formatter.use_locale'] = True
    #Create grid
    gs = GridSpec(5, 5, figure=fig)
    #Empty list for axes
    ax = [[None for _ in range(5)] for _ in range(6)]
    #Maximum value of histograms
    max_counts = []
    #Axis limits
    lim = []
    #1D plots
    for i in (0,1,2,3,4):
        #Create subplot
        ax[i][i] = fig.add_subplot(gs[i,i])
        #Plot
        h = HistSimple(ax[i][i],samples[:,i])
        #Define the function
        a,loc,scale = skew_norm_params[i]
        skn = skewnorm(a, loc, scale)
        #Get max value
        max_counts+=[max(h[0])]
        #Plot limits
        ax[i][i].set_ylim(min(h[1]),max(h[1]))
        lim += [[min(h[1]),max(h[1])]]#ax[i][i].get_ylim()
        #Get x
        y = linspace(min(h[1]),max(h[1]),1000)
        #Get y
        if (i==1) & (age_plot=='linear'):
            prior = exp([ priors[i](yy,inputs['Priors'][indexes[i]]['Parameters']) for yy in log10(y)+9 ] )
        elif (i==2) & (distance_plot=='modulus'):
            prior = exp([ priors[i](yy,inputs['Priors'][indexes[i]]['Parameters']) for yy in 10**( (y+5)/5-3 ) ] )
        else:
            prior = exp([ priors[i](yy,inputs['Priors'][indexes[i]]['Parameters']) for yy in y ] )
        pdf = skn.pdf(y)
        #Plot prior
        ax[i][i].plot(prior/max(prior)*max_counts[i],y,ls=':',c='k')
        #Plot fitted skewnorm
        ax[i][i].plot( pdf /max(pdf)*max_counts[i],y,ls='-',c='tab:blue',zorder=-1)
        #Span
        ax[i][i].errorbar( max_counts[i]/2, fit_values_plot[i],
                           yerr = fit_uncertainty[i],
                           fmt='o',c='tab:red',capsize=3,elinewidth=1,markersize=3)
        
        if i<4:ax[i][i].tick_params(labelleft=False)
        #Side
        ax[i][i].yaxis.tick_right()
        ax[i][i].yaxis.set_label_position("right")
        ax[i][i].xaxis.tick_top()
        ax[i][i].xaxis.set_label_position("top")
        ax[i][i].set_xlim(ax[i][i].get_xlim()[::-1])
        #Ticks
        ax[i][i].tick_params(axis="both",which='both',direction="in")
    ax[4][4].set_ylabel(labels[4])
    #2D plots
    for j in (1,2,3,4):
        for i in range(0,j):
            #Create subplot
            ax[i][j] =  fig.add_subplot(gs[i,j])
            #Plot
            Hist2D(ax[i][j],samples[:,j],samples[:,i],
                   [best_values[j],best_values[i]],
                   [median_values[j],median_values[i]])
            #Labels
            if j == 4: ax[i][j].set_ylabel(labels[i])
            else: ax[i][j].tick_params(labelleft=False)
            if i == 0: ax[i][j].set_xlabel(labels[j])
            else: ax[i][j].tick_params(labelbottom=False)
            #Scatter
            ax[i][j].errorbar( fit_values_plot[j], fit_values_plot[i],
                               xerr = fit_uncertainty[j],
                               yerr = fit_uncertainty[i],
                               fmt='o',c='tab:red',capsize=3,elinewidth=1,markersize=3)
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
            #ax[i][j].get_shared_y_axes().join(ax[i][j], ax[i][i]) ; ax[i][j].autoscale()
    #Axis details
    '''
    for i in (0,1,2,3,4):
        #Share axis
        ax[i][i].get_shared_y_axes().join(ax[i][i], ax[i][4]) ; ax[i][i].autoscale()
        '''
    #Iterate over plots       
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
    plot_text = '$[M/H]=${:.2f} $\pm$ {:.2f}\n'.format(fit_values_plot[0],fit_uncertainty[0]) 
    if age_plot == 'linear':
        plot_text += 'Idade={:.2f} $\pm$ {:.2f} Ganos \n'.format(fit_values_plot[1],fit_uncertainty[1])
    else: 
        plot_text += '$\log Idade_{{Ganos}}=${:.2f} $\pm$ {:.2f}\n'.format(fit_values_plot[1],fit_uncertainty[1])
    if distance_plot == 'modulus':
        plot_text += '$(m-M)_0=${:.2f} $\pm$ {:.2f}\n'.format(fit_values_plot[2],fit_uncertainty[2])
    else:
        plot_text += '$d=${:.2f} $\pm$ {:.2f} kpc \n'.format(fit_values_plot[2],fit_uncertainty[2])
    plot_text += '$E(B-V)=${:.2f} $\pm$ {:.2f}\n'.format(fit_values_plot[3],fit_uncertainty[3]) 
    plot_text +='Frac bin$=${:.2f} $\pm$ {:.2f}'.format(fit_values_plot[4],fit_uncertainty[4])
    axtext.text(0.0,0.5,plot_text.replace('.',','),
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
    # ADD CMD
    #
    import SIESTAmodules
    from pandas import DataFrame,read_csv
    gsCMD = GridSpecFromSubplotSpec(1, 2, subplot_spec = gs[3:,0:3])
    #Subplots
    axCMD1 = fig.add_subplot(gsCMD[0])
    axCMD2 = fig.add_subplot(gsCMD[1])
    #Ticks
    axCMD2.tick_params(labelleft=False)
    axCMD1.tick_params(axis="both",direction="in")
    axCMD2.tick_params(axis="both",direction="in")
    #Get cluster CMD
    clusterCMD = read_csv('projects/{}/FilteredCMD.dat'.format(project), sep=',')
    #Get isochrone index
    isochroneIndex,_,_ = SIESTAmodules.Initialization.GetIsochroneIndex(inputs['Grid_path'])
    #Isochrone
    mh = round(fit_values[0],2)
    age = round(fit_values[1],2)
    iso = read_csv('{}/{}.dat'.format(inputs['Grid_path'],isochroneIndex[mh][age]),sep=',')
    #Displace isochrone
    DM_V,DM_I,E_VI = SIESTAmodules.Auxiliary.CMDshift(fit_values[2],fit_values[3])
    iso['Vapp'] = iso['Vmag'] + DM_V
    iso['Iapp'] = iso['Imag'] + DM_I
    isomini = iso[iso['Vapp'] <= inputs['Photometric_limit']]
    #Random numbers for synthetic population
    PopulationSamplesRN, PhotometricErrorRN = SIESTAmodules.Initialization.RandomNumberStarter(inputs['Initial_population_size'],inputs['Seed'])
    #Synthetic population
    syntCMD = DataFrame()
    SIESTAmodules.SyntheticPopulation.Generator(syntCMD,
                                                 iso.copy(), 
                                                 inputs['Initial_population_size'],
                                                 fit_values[4], 
                                                 inputs['Companion_min_mass_fraction'],
                                                 fit_values[2],fit_values[3],
                                                 inputs['Photometric_limit'],
                                                 inputs['Error_coefficients'], 
                                                 inputs['Completeness_Fermi'],
                                                 PopulationSamplesRN, PhotometricErrorRN)
    syntCMD = syntCMD.sample(len(clusterCMD),random_state=seed)
    #FIRST PLOT
    axCMD1.scatter(clusterCMD['V-I'],clusterCMD['V'],c='tab:orange',marker='.',label='Cluster',rasterized=True)
    axCMD1.scatter(syntCMD['V']-syntCMD['I'],syntCMD['V'],c='tab:green',marker='.',label='Synt. pop.',rasterized=True)
    axCMD1.plot(isomini['Vapp']-isomini['Iapp'],isomini['Vapp'],c='k',ls='--')
    axCMD1.set_ylabel(r'$V$')
    axCMD1.invert_yaxis()
    #axCMD1.legend()
    #SECOND PLOT
    axCMD2.scatter(syntCMD['V']-syntCMD['I'],syntCMD['V'],c='tab:green',marker='.',label='Synt. pop.',rasterized=True)
    axCMD2.scatter(clusterCMD['V-I'],clusterCMD['V'],c='tab:orange',marker='.',label='Cluster',rasterized=True)
    axCMD2.plot(isomini['Vapp']-isomini['Iapp'],isomini['Vapp'],c='k',ls='--')
    #axCMD2.legend()
    axCMD2.get_shared_y_axes().join(axCMD1, axCMD2) ; axCMD2.autoscale()
    #SHARED DETAILS
    for ax in (axCMD1,axCMD2):
        ax.set_xlabel(r'$V-I$')
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
    #
    #
    #Return
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    return best_values, median_values, fit_values, fig,samples

def Population(project,values,seed):
    #Import libraries
    from numpy import abs, linspace,loadtxt,log10
    from pickle import load
    import SIESTAmodules
    from pandas import read_csv,DataFrame
    from sklearn.neighbors import NearestNeighbors
    from matplotlib.gridspec import GridSpec
    #Open inputs
    with open('projects/{}/inputs.pkl'.format(project),'rb') as pkl_file: inputs = load(pkl_file)
    #Get cluster CMD
    clusterCMD = read_csv('projects/{}/FilteredCMD.dat'.format(project), sep=',')
    #Cluster grid
    ClusterGrid = loadtxt( 'projects/{}/ClusterGrid.dat'.format(inputs['Project_name']))
    #Inputs
    #
    # PLOTS
    #
    #Get isochrone index
    isochroneIndex,_,_ = SIESTAmodules.Initialization.GetIsochroneIndex(inputs['Grid_path'])
    #Create figure
    fig = plt.figure(constrained_layout=True,figsize=(9,7))
    #plt.rcParams['font.size'] = '14'
    #Create grid
    gs = GridSpec(3, 6, figure=fig)
    #Isochrone
    mh = round(values[0],2)
    age = round(values[1],2)
    iso = read_csv('{}/{}.dat'.format(inputs['Grid_path'],isochroneIndex[mh][age]),sep=',')
    #Displace isochrone
    DM_V,DM_I,E_VI = SIESTAmodules.Auxiliary.CMDshift(values[2],values[3])
    iso['Vapp'] = iso['Vmag'] + DM_V
    iso['Iapp'] = iso['Imag'] + DM_I
    isomini = iso[iso['Vapp'] <= inputs['Photometric_limit']]
    #Random numbers for synthetic population
    PopulationSamplesRN, PhotometricErrorRN = SIESTAmodules.Initialization.RandomNumberStarter(inputs['Initial_population_size'],inputs['Seed'])
    #Synthetic population
    syntCMD = DataFrame()
    SIESTAmodules.SyntheticPopulation.Generator(syntCMD,
                                                 iso.copy(), 
                                                 inputs['Initial_population_size'],
                                                 values[4], 
                                                 inputs['Companion_min_mass_fraction'],
                                                 values[2],values[3],
                                                 inputs['Photometric_limit'],
                                                 inputs['Error_coefficients'], 
                                                 inputs['Completeness_Fermi'],
                                                 PopulationSamplesRN, PhotometricErrorRN)
    #Densities
    SyntheticGrid = SIESTAmodules.Distribution.Evaluate(inputs['Edges_color'], inputs['Edges_magnitude'],
                                                         syntCMD['Vfilled']-syntCMD['Ifilled'], syntCMD['Vfilled'],
                                                         renorm_factor=inputs['Cluster_size']/len(syntCMD))
    
    syntCMD = syntCMD.sample(n=1*len(clusterCMD),random_state=seed)#inputs['Seed'])
    #
    # FIRST PLOT
    #
    ax =  fig.add_subplot(gs[0:2,0:3])
    ax.scatter(clusterCMD['V-I'],clusterCMD['V'],c='tab:orange',marker='$\u25EF$',label='Cluster')
    ax.scatter(syntCMD['V']-syntCMD['I'],syntCMD['V'],c='tab:green',marker='$\u25EF$',label='Synt. pop.')
    ax.plot(isomini['Vapp']-isomini['Iapp'],isomini['Vapp'],c='k',ls='--')
    ax.legend()
    ax.set_xlabel(r'$V-I$')
    ax.set_ylabel(r'$V$')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.invert_yaxis()
    #
    # SECOND PLOT
    #
    ax =  fig.add_subplot(gs[0:2,3:])
    ax.scatter(syntCMD['V']-syntCMD['I'],syntCMD['V'],c='tab:green',marker='$\u25EF$',label='Synt. pop.')
    ax.scatter(clusterCMD['V-I'],clusterCMD['V'],c='tab:orange',marker='$\u25EF$',label='Cluster')
    ax.plot(isomini['Vapp']-isomini['Iapp'],isomini['Vapp'],c='k',ls='--')
    ax.legend()
    ax.set_xlabel(r'$V-I$')
    ax.set_ylabel(r'$V$')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.invert_yaxis()
    #
    # THIRD PLOT
    #
    ax =  fig.add_subplot(gs[2,0:2])
    #Color limits
    cmin = min( [ ClusterGrid.min(), SyntheticGrid.min()] )
    cmax = max( [ ClusterGrid.max(), SyntheticGrid.max()] )
    #Import some inputs
    colors = inputs['Edges_color']
    magnitudes = inputs['Edges_magnitude']
    #Density map
    from matplotlib.colors import LogNorm
    '''
    hist = ax.imshow( ClusterGrid, cmap='Blues', norm=LogNorm(vmin=1e-1, vmax=cmax), 
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
    
    '''
    hist = ax.imshow( ClusterGrid, cmap='Oranges', vmin = cmin, vmax = cmax , 
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
                         
    #Colorbar
    fig.colorbar(hist,ax=ax,orientation='horizontal',label='Counts')
    ax.set_xlabel(r'$v-i$')
    ax.set_ylabel(r'$v$')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #
    # FOURTH PLOT
    #
    ax =  fig.add_subplot(gs[2,2:4])
    from matplotlib.colors import LogNorm
    '''
    hist = ax.imshow( SyntheticGrid, cmap='Reds', norm=LogNorm(vmin=1e-1, vmax=cmax), 
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
                         '''
    #Density map
    
    hist = ax.imshow( SyntheticGrid, cmap='Greens', vmin = cmin, vmax = cmax,
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
    #Colorbar
    fig.colorbar(hist,ax=ax,orientation='horizontal',label='Counts')
    ax.set_xlabel(r'$v-i$')
    ax.set_ylabel(r'$v$')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #
    # FIFTH PLOT
    #
    from numpy import log    
    from scipy.special import loggamma 
    ax =  fig.add_subplot(gs[2,4:])
    #Likelihood
    LogLike = loggamma( 0.5 + ClusterGrid + SyntheticGrid ) - ( 0.5 + ClusterGrid + SyntheticGrid )*log(2) - loggamma(0.5 + SyntheticGrid) - loggamma( 1 + ClusterGrid )
    #Density map
    hist = ax.imshow( LogLike , cmap='Purples',
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
    #Colorbar
    fig.colorbar(hist,ax=ax,orientation='horizontal',label=r'$\log L$')
    '''
    #Density difference 
    DiffGrid = ClusterGrid - SyntheticGrid
    #Color limits
    clim = max( [-DiffGrid.min(),DiffGrid.max()] )
    #Density map
    hist = ax.imshow( DiffGrid , cmap='RdBu', vmin = -clim, vmax= clim,
                         extent = [colors[0],colors[-1],magnitudes[-1],magnitudes[0]],aspect='auto')
    #Colorbar
    fig.colorbar(hist,ax=ax,orientation='horizontal',label='Counts')
    '''
    ax.set_xlabel(r'$v-i$')
    ax.set_ylabel(r'$v$')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #Return figure
    return fig

def SaveAll(project_name,figures):
     #Import libraries
    from matplotlib.backends.backend_pdf import PdfPages
    #Path name
    path = 'projects/{}'.format(project_name)
    #Image pdf
    pdf = PdfPages('{}/results.pdf'.format(path))
    #Save figures
    for fig in figures:
        pdf.savefig(fig)
    pdf.close()
        