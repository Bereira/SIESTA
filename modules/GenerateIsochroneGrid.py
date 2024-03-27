#
# Create isochrone grid
#

def Run(DownloadedModelsPath, IsochronesPath, HeaderSize, FilterColumns, ClipLateEvolution):
    #Import specific libraries
    from glob import glob
    import os
    import pandas as pd
    from collections import defaultdict
    from tqdm import tqdm
    import pickle
    #Create output file, if it does not exists
    if not os.path.exists(IsochronesPath):
        os.makedirs(IsochronesPath)
    #Read the names of all downloaded files
    files = glob(DownloadedModelsPath+'/*')
    #Create index
    index = defaultdict(dict)
    #Iterate over files 
    print('Iterating over isochrones...')
    for file in tqdm(files):
        #Read file header
        with open(file) as f: header = [next(f) for x in range(0,HeaderSize)]
        #Get table column names
        names = header[-1].replace('#','').replace('\n','').split()
        #Import tables
        data = pd.read_csv(file,names=names,sep='\s+',comment='#')
        #Get unique metallicity and ages
        MH_list = data['MH'].unique()
        Age_list = data['logAge'].unique()
        #Iterate over metallicities and ages
        for mh in MH_list:
            for logAge in Age_list:
                #Select data from specific isochrone
                data_iso = data[ (data['MH'] == mh) & (data['logAge'] == logAge)]
                #Select columns
                if FilterColumns != 'no':
                    data_iso = data_iso.loc[:,FilterColumns]
                #Select rows
                if ClipLateEvolution == 'yes':
                    data_iso = data_iso[data_iso['label']<=7]           
                #File name
                filename = 'MH{:.2f}_logAge{:.2f}'.format(mh,logAge).replace('-','n').replace('.','d')
                #Save file
                data_iso.reset_index(drop=True).to_csv('{}/{}.dat'.format(IsochronesPath,filename))
                index[mh][logAge] = filename
    #Save index
    with open('{}/index.pkl'.format(IsochronesPath), "wb") as out_file:
        pickle.dump(index, out_file)
    #Interact
    print('All done! Check {} directory'.format(IsochronesPath))

            
        