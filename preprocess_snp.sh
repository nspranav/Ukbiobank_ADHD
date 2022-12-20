filepath_analysis=/data/analysis/collaboration/Multi-sites_genetics/UKB/genetics
filepath_ukbiobank=/data/mialab/competition2019/UKBiobank
filepath_ukbiobank_genetics=${filepath_ukbiobank}/genetics/data



######################################################################################
##### genotype data QC ###############################################################
######################################################################################

filepath1=${filepath_analysis}/geno_IMGsamples
cd ${filepath1}
filename=ukb_cal_imggen

# check heterozygosity
${plinkpath}/plink --bfile ${filename} --het --noweb --out ${filename} 

# exclude samples with high heterozygosity ratio (in general 3SD, but depending on the data sometimes may need to use different criteria) 
${plinkpath}/plink --bfile ${filename} --remove samplelist_hetero.txt --make-bed --noweb --out ${filename}_1 

# snp missing ratio, MAF and HWE control
${plinkpath}/plink --bfile ${filename}_1 --geno 0.05 --maf 0.01 --hwe 1e-06 --noweb --make-bed --out ${filename}_1

# sample missing ratio
${plinkpath}/plink --bfile ${filename}_1 --missing --noweb --out ${filename}_1

# exclude samples with high missing ratio 
${plinkpath}/plink --bfile ${filename}_1 --remove samplelist_imiss.txt --noweb --make-bed --out ${filename}_2

# IBD (for genotype data you can do this before pruning) and nearest neighbor
${plinkpath}/plink --bfile ${filename}_2 --genome --noweb --out ${filename}_2
${plinkpath}/plink --bfile ${filename}_2 --read-genome ${filename}_2.genome --cluster --out ${filename}_2
awk '$10 >= 0.1875 {print $1,$2,$3,$4,$10}' ${filename}_2.genome > samplelist_ibd.csv

# exclude relatives after checking IBD data
${plinkpath}/plink --bfile ${filename}_2 --remove samplelist_ibd_ex.txt --make-bed --out ${filename}_3

# check non-random missings
${plinkpath}/plink --bfile ${filename}_3 --test-mishap --noweb --out ${filename}_3

# exclude SNPs with non-random missings
${plinkpath}/plink --bfile ${filename}_3 --exclude snplist_mishap.txt --make-bed --noweb --out ${filename}_3