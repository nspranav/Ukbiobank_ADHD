#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
# %%

path = '/data/users2/pnadigapusuresh1/Projects/ukbiobank/Data/'
chr1_bim = pd.read_csv(path+'chr1.bim',sep='\t',header=None)

# %%
p_seg = chr1_bim[chr1_bim[3] < 125000000][[1,4]]
p_s = p_seg[1] + '_' + p_seg[4]
q_seg = chr1_bim[chr1_bim[3] >= 125000000][[1,4]]
q_s = q_seg[1] + '_' + q_seg[4]

# %%
chr1 = pd.read_csv(path+'chr1_recode.raw',sep=' ')
chr1_p = chr1[p_s]
chr1_p_imp = SimpleImputer(strategy='most_frequent').fit_transform(chr1_p)
chr1_q = chr1[q_s]
chr1_q_imp = SimpleImputer(strategy='most_frequent').fit_transform(chr1_q)
#%%
chr1_scaled = StandardScaler().fit_transform(chr1_p_imp)
chr1_q_scaled = StandardScaler().fit_transform(chr1_q_imp)
#%%
pca = PCA(n_components=0.8)
chr1_p_pca = pca.fit_transform(chr1_scaled)
chr1_q_pca = pca.fit_transform(chr1_q_scaled)

#%%

pos = {1:125000000,2:96800000,3:91000000,4:50400000,5:48400000,6:61000000,7:59900000,
8:45600000,10:42300000,11:53700000}
pca = PCA(n_components=0.8)
for k in range(1,23):
    try:
        # chr_bim = pd.read_csv(path+'chr'+str(k)+'.bim',sep='\t',header=None)
        
        # p_seg = chr_bim[chr_bim[3] < v][[1,4]]
        # p_s = p_seg[1] + '_' + p_seg[4]
        # q_seg = chr_bim[chr_bim[3] >= v][[1,4]]
        # q_s = q_seg[1] + '_' + q_seg[4]

        chr = pd.read_csv(path+'chr' + str(k) + '_recode.raw',sep=' ')
        chr = chr.iloc[:,6:]
        chr_imp = SimpleImputer(strategy='most_frequent').fit_transform(chr)
        chr_scaled = StandardScaler().fit_transform(chr_imp)
        chr_pca = pca.fit_transform(chr_scaled)
        
        # chr_p = chr[p_s]
        # chr_p_imp = SimpleImputer(strategy='most_frequent').fit_transform(chr_p)
        # chr_q = chr[q_s]
        # chr_q_imp = SimpleImputer(strategy='most_frequent').fit_transform(chr_q)

        # chr_p_scaled = StandardScaler().fit_transform(chr_p_imp)
        # chr_q_scaled = StandardScaler().fit_transform(chr_q_imp)

        # chr_p_pca = pca.fit_transform(chr_p_scaled)

        # with open(path+'PCA/chr'+str(k) + '_p.pca','wb') as f:
        #     pickle.dump(pca,f)

        # with open(path+'PCA_transform/chr'+ str(k) +'_p.comp','wb') as f:
        #     pickle.dump(chr_p_pca,f)
        
        # chr_q_pca = pca.fit_transform(chr_q_scaled)

        # with open(path+'PCA/chr'+str(k) + '_q.pca','wb') as f:
        #     pickle.dump(pca,f)

        # with open(path+'PCA_transform/chr'+ str(k) +'_q.comp','wb') as f:
        #     pickle.dump(chr_q_pca,f)


        with open(path+'PCA/chr'+str(k) + '.pca','wb') as f:
            pickle.dump(pca,f)

        with open(path+'PCA_transform/chr'+ str(k) +'.comp','wb') as f:
            pickle.dump(chr_pca,f)
        
    except Exception as e:
        print(f'chr{k} did not work, with exception {e}')
        


# %%
