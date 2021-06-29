#%%
import subprocess

path = '/data/users2/pnadigapusuresh1/Projects/ukbiobank/Data/'
#%%
for i in range(1,23):
    #subprocess.run(['plink', '--bfile', path+'ukb_cal_imggen_5', '--chr' ,str(i),'--make-bed', '--out', path+ '/chr'+str(i)])
    #subprocess.run(['plink', '--bfile', 'Data/chr_'+str(i), '--geno',str(0.01),'--make-bed', '--out','Data/geno_001/chr_'+str(i)])
    subprocess.run(['plink','--bfile', path+'chr'+str(i),'--recode','A','--out',path+'chr'+str(i)+'_recode'])
    #subprocess.run(['ls'])

#%%

