import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

if __name__ == "__main__":
    path = '/data/users2/pnadigapusuresh1/Projects/ukbiobank/Data'
    
    for i in range(1,2):
        print('started')
        chr = pd.read_csv(path+'/chr_'+str(i)+'_recode.raw',sep=' ')
        print('Finished reading')
        chr_allele = chr.iloc[:,6:]
        imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        print('Imputing')
        chr_allele_imputed = imp_mode.fit_transform(chr_allele)
        chr_allele_imputed = StandardScaler().fit_transform(chr_allele_imputed)
        print('Performing PCA')
        pca = PCA(n_components=0.8)
        pca.fit(chr_allele_imputed)
        print('Saving pca')
        with open(path+'/PCA/chr'+str(i)+'.pca','wb') as f:
            pickle.dump(pca,f)
        
        chr_pca = pca.transform(chr_allele_imputed)
        print(f'shape of pca is {chr_pca.shape}')
        print('saving pca transform')
        with open(path+'/PCA_transform/chr'+str(i)+'.vec','wb') as f:
            pickle.dump(chr_pca,f)