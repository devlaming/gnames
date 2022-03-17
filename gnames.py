import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import t

class gnames:
    '''
    Genetic-Nurture and Assortative-Mating-Effects Simulator (GNAMES)
    
    GNAMES can be used to quickly generate many generations of genotype and
    phenotype data under any desired level of assortative mating and under the
    presence of genetic-nurture effects, allowing for various follow-up
    analyses, such as GWAS, polygenic prediction, GREML, and ORIV, permitting
    users to control for family background
    
    Author: Ronald de Vlaming
    Repository: https://github.com/devlaming/gnames
    
    Attributes
    ----------
    iN : int > 1
        number of unrelated founders
    
    iM : int > 0
        number of biallelic, autosomal SNPs
    
    iC : int > 0, optional
        number of children per mating pair; default=2
    
    dHsqY : float in [0,1], optional
        heritability of main trait Y; default=0.5
    
    dPropGN : float in [0,1], optional
        proportion of variance of Y accounted for by genetetic nurture (GN);
        default=0.25
    
    dCorrYAM : float in [-1,+1], optional
        correlation of assortative-mating (AM) trait and Y (uncorrelated
        part drawn as Gaussian noise); default=1
    
    dRhoAM : float in [-1,+1], optional
        AM strenght = correlation in AM trait between mates;
        default=0.5
    
    dRhoSibE : float in [-1,+1], optional
        environment correlation of Y across siblings; default=0
    
    iSF : int >= 0, optional
        block size of families when generating genotypes and phenotypes;
        default=0, which is treated as having one big block for all families
    
    iSM : int >= 0, optional
        block size of SNPs when generating genotypes and phenotypes;
        default=0, which is treated as having one big block for all SNPs
    
    dBetaAF0 : float > 0, optional
        single value for the two shape parameters of a beta distribution
        used to draw SNP allele frequencies of founders; default=0.35
        
    dMAF0 : float in (0,0.45), optional
        minor-allele frequency threshold imposed when drawing allele
        frequencies from beta distribution; default=0.1
    
    iSeed : int > 0, optional
        seed for random-number generator used by GNAMES (for replicability);
        default=502421368
    
    Methods
    -------
    Simulate(iGenerations=1)
        Simulate data for a given number of new generations, under assortative
        mating of parents and simulating offspring genotypes and phenotypes
    
    ComputeDiagsGRM(dMAF=0.01)
        Compute diagonal elements of the GRM for the current generation,
        excluding SNPs with a minor allele frequency below the given threshold
    
    MakeGRM(sName='genotypes',dMAF=0.01,vFamInd=None)
        Make GRM in GCTA binary format for given set of families (default=all)
    
    MakeBed(sName='genotypes')
        Export genotypes and phenotypes to PLINK files
    
    PerformGWAS(sName='results')
        Perform classical GWAS and within-family GWAS based on offspring data
    
    MakeThreePGIs(sName='results',iNGWAS=None,iNPGI=None)
        Perform 2 GWASs on non-overlapping samples, considering 1 child per
        family. Also perform a GWAS on these 2 GWAS samples pooled. Use these
        3 sets of GWAS estimates to construct 3 PGIs in the hold-out sample.
        For the hold-out sample, export these PGIs, the GRM, and phenotype.
    '''
    dTooHighMAFThreshold=0.45
    tIDs=('FID','IID')
    sBedExt='.bed'
    sBimExt='.bim'
    sFamExt='.fam'
    sPheExt='.phe'
    iMale=1
    iFemale=2
    iMissY=-9
    sGrmBinExt='.grm.bin'
    sGrmBinNExt='.grm.N.bin'
    sGrmIdExt='.grm.id'
    sGWASExt='.GWAS.classical.txt'
    sWFExt='.GWAS.within_family.txt'
    sPGIExt='.pgi'
    sINFOExt='.info'
    binBED1=bytes([0b01101100])
    binBED2=bytes([0b00011011])
    binBED3=bytes([0b00000001])
    iNperByte=4
    lAlleles=['A','C','G','T']
    lGWAScol=['Baseline Allele','Effect Allele','Per-allele effect estimate',\
               'Standard error','T-test statistic','P-value',\
                   'Effect Allele Frequency']
    sPGIheader='FID\tIID\tPID\tMID\tY\tG\tE\tN\tAM\tPGI True\tPGI GWAS 1\t'+\
        'PGI GWAS 2\tPGI GWAS Pooled'
    dPropGWAS=0.4
    dPropPGI=0.2
    def __init__(self,iN,iM,iC=2,dHsqY=0.5,dPropGN=0.25,dCorrYAM=1,\
                 dRhoAM=0.5,dRhoSibE=0,iSF=0,iSM=0,\
                 dBetaAF0=0.35,dMAF0=0.1,iSeed=502421368):
        if not(isinstance(iN,int)):
            raise ValueError('Number of founders not integer')
        if not(isinstance(iM,int)):
            raise ValueError('Number of SNPs not integer')
        if not(isinstance(iC,int)):
            raise ValueError('Number of children not integer')
        if not(isinstance(dHsqY,(int,float))):
            raise ValueError('Heritability of main trait Y is not a number')
        if not(isinstance(dPropGN,(int,float))):
            raise ValueError('Proportion of variance in Y accounted for by'+\
                             ' genetic nurture is not a number')
        if not(isinstance(dCorrYAM,(int,float))):
            raise ValueError('Correlation of assortative-mating trait and Y'+\
                             ' is not a number')
        if not(isinstance(dRhoAM,(int,float))):
            raise ValueError('Degree of assortative mating is not a number')
        if not(isinstance(dRhoSibE,(int,float))):
            raise ValueError('Environment correlation of Y across siblings'+\
                             ' not a number')
        if not(isinstance(iSF,int)):
            raise ValueError('Block size for families not integer')
        if not(isinstance(iSM,int)):
            raise ValueError('Block size for SNPs not integer')
        if not(isinstance(dBetaAF0,(int,float))):
            raise ValueError('Parameter of beta distribution used to draw'+\
                             ' allele frequencies not a number')
        if not(isinstance(dMAF0,(int,float))):
            raise ValueError('Minor-allele-frequency threshold not a number')
        if not(isinstance(iSeed,int)):
            raise ValueError('Seed for random-number generator not integer')
        if iN<2:
            raise ValueError('Number of founders less than two')
        if iM<1:
            raise ValueError('Number of SNPs less than one')
        if iC<1:
            raise ValueError('Number of children less than one')
        if dHsqY>1 or dHsqY<0:
            raise ValueError('Heritability of main trait Y is'+\
                             ' not constrained to [0,1] interval')
        if dPropGN>1 or dPropGN<0:
            raise ValueError('Proportion of variance in Y accounted for by'+\
                             'genetic nurture is not constrained to [0,1]'+\
                             ' interval')
        if (dHsqY+dPropGN)>1:
            raise ValueError('Heritability and genetic nurture combined'+\
                             ' explain more than 100% of the variance in Y')
        if dCorrYAM>1 or dCorrYAM<-1:
            raise ValueError('Correlation of assortative-mating trait and Y'+\
                             ' is not constrained to [-1,+1] interval')
        if dRhoAM>1 or dRhoAM<-1:
            raise ValueError('Degree of assortative mating is not'+\
                             ' not constrained to [-1,+1] interval')
        if dRhoSibE>1 or dRhoSibE<0:
            raise ValueError('Environment correlation of Y across siblings'+\
                             ' is not constrained to [0,1] interval')
        if iSF<0:
            raise ValueError('Block size for families negative')
        if iSM<0:
            raise ValueError('Block size for SNPs negative')
        if dBetaAF0<=0:
            raise ValueError('Parameter for beta distribution to draw'+\
                             ' allele frequencies is non-positive')
        if dMAF0<=0:
            raise ValueError('Minor-allele-frequency threshold is'+\
                             ' non-positive')
        if dMAF0>=gnames.dTooHighMAFThreshold:
            raise ValueError('Minor-allele-frequency threshold is'+\
                             ' unreasonably high')
        if iSeed<0:
            raise ValueError('Seed for random-number generator negative')
        self.iP=iN
        self.iC=iC
        self.iM=iM
        self.iSF=iSF
        self.iSM=iSM
        self.dHsqY=dHsqY
        self.dPropGN=dPropGN       
        self.dCorrYAM=dCorrYAM       
        self.dRhoAM=dRhoAM
        self.dRhoSibE=dRhoSibE
        self.dBetaAF0=dBetaAF0
        self.dMAF0=dMAF0
        self.rng=np.random.RandomState(iSeed)
        self.__set_loadings_rho_sib_e()
        self.__draw_alleles()
        self.__draw_afs()
        self.__draw_betas()
        self.__draw_gen0()
    
    def Simulate(self,iGenerations=1):
        """
        Simulate data for a given number of new generations
        
        Simulates offspring genotypes and phenotypes under assortative mating
        of parents, where main phenotype Y is subject to genetic nurture,
        and AM is the assortative-mating trait
        
        Attributes
        ----------
        iGenerations : int > 0, optional
            number of new generations to simulate data for; default=1
        """
        if not(isinstance(iGenerations,int)):
            raise ValueError('Number of generations not integer')
        if iGenerations<1:
            raise ValueError('Number of generations non-positive')
        for i in tqdm(range(iGenerations)):
            self.__draw_next_gen()
    
    def __set_loadings_rho_sib_e(self):
        mRhoSibE=(1-self.dRhoSibE)*np.eye(self.iC)\
            +self.dRhoSibE*np.ones((self.iC,self.iC))
        (vD,mP)=np.linalg.eigh(mRhoSibE)
        vD[vD<=0]=0
        self.mWeightSibE=(mP*((vD**0.5)[None,:]))
    
    def __draw_alleles(self):
        print('Drawing SNP alleles')
        self.vChr=np.zeros(self.iM,dtype=np.uint8)
        self.lSNPs=['rs'+str(i+1) for i in range(self.iM)]
        lA1A2=[self.rng.choice(gnames.lAlleles,size=2,\
                               replace=False) for i in range(self.iM)]
        mA1A2=np.array(lA1A2)
        self.vA1=mA1A2[:,0]
        self.vA2=mA1A2[:,1]
        self.vCM=np.zeros(self.iM,dtype=np.uint8)
        self.vPos=np.arange(self.iM)+1
    
    def __draw_afs(self):
        print('Drawing allele frequencies')
        vAF=self.rng.beta(self.dBetaAF0,self.dBetaAF0,self.iM)
        while (min(vAF)<self.dMAF0)|(max(vAF)>(1-self.dMAF0)):
            vAF[vAF<self.dMAF0]=\
                self.rng.beta(self.dBetaAF0,\
                              self.dBetaAF0,np.sum(vAF<self.dMAF0))
            vAF[vAF>(1-self.dMAF0)]=\
                self.rng.beta(self.dBetaAF0,\
                              self.dBetaAF0,np.sum(vAF>(1-self.dMAF0)))
        self.vAF0=vAF
        self.vTau0=(1-vAF)**2
        self.vTau1=1-(vAF**2)
    
    def __draw_betas(self):
        print('Drawing true SNP effects')
        vScaling=(2*self.vAF0*(1-self.vAF0))**(-0.5)
        self.vBeta=self.rng.normal(size=self.iM)*vScaling
        self.vGamma=self.rng.normal(size=self.iM)*vScaling
    
    def __draw_gen0(self):
        self.iT=0
        self.__set_counts_ids_dims()
        self.__set_chunks()
        self.__draw_g0()
        self.__draw_y()
    
    def __draw_next_gen(self):
        self.iT+=1
        self.__match()
        self.__set_counts_ids_dims()
        self.__set_chunks()
        self.__mate()
        self.__draw_y()
    
    def __set_counts_ids_dims(self):
        if self.iT==0:
            self.vPID=np.zeros(self.iP,dtype=np.uint32)
            self.vMID=np.zeros(self.iP,dtype=np.uint32)
            self.vGN=self.rng.normal(size=self.iP)
            self.iNT=0
            self.iS=1
        else:
            self.iS=self.iC
        self.iF=self.iP
        self.iN=self.iS*self.iF
        self.iPS=int(self.iF/2)
        self.iP=self.iS*self.iPS
        self.mFAM=np.zeros((self.iS,self.iF,5),dtype=np.uint32)
        self.mFAM[:,:,1]=(self.iNT+np.arange(self.iN)).\
            reshape((self.iS,self.iF))
        for h in range(self.iS):
            self.mFAM[h,:,2]=self.vPID
            self.mFAM[h,:,3]=self.vMID
            vRowInd=self.rng.permutation(self.iF)
            vRowIndFemale=vRowInd[0:self.iPS]
            vRowIndMale=vRowInd[self.iPS:2*self.iPS]
            self.mFAM[h,vRowIndFemale,4]=gnames.iFemale
            self.mFAM[h,vRowIndMale,4]=gnames.iMale
        self.iNT+=self.iN
    
    def __set_chunks(self):
        if self.iSF==0:
            self.iSFT=self.iF
        else:
            self.iSFT=self.iSF
        if self.iSM==0:
            self.iSM=self.iM
        self.iFB=int(self.iF/self.iSFT)
        self.iFR=(self.iF)%(self.iSFT)
        self.iFT=self.iFB+(self.iFR>0)
        self.iMB=int(self.iM/self.iSM)
        self.iMR=(self.iM)%(self.iSM)
        self.iMT=self.iMB+(self.iMR>0)
        self.iChunks=self.iFT*self.iMT
    
    def __draw_g0(self):
        print('Drawing genotypes unrelated founders')
        self.mG=np.empty((self.iS,self.iF,self.iM),dtype=np.uint8)
        if self.iChunks>1: tCount=tqdm(total=self.iChunks)
        for j in range(self.iMT):
            iM0=self.iSM*j
            iM1=min(self.iM,iM0+self.iSM)
            vTau0=self.vTau0[iM0:iM1]
            vTau1=self.vTau1[iM0:iM1]
            for i in range(self.iFT):
                iF0=self.iSFT*i
                iF1=min(self.iF,iF0+self.iSFT)
                self.__draw_g0_chunk(iF0,iF1,iM0,iM1,vTau0,vTau1)
                if self.iChunks>1: tCount.update(1)
        if self.iChunks>1: tCount.close()
    
    def __draw_g0_chunk(self,iF0,iF1,iM0,iM1,vTau0,vTau1):
        iF=iF1-iF0
        iM=iM1-iM0
        mU=self.rng.uniform(size=(iF,iM))
        mG=np.ones((iF,iM),dtype=np.uint8)
        mG[mU<(vTau0[None,:])]=0
        mG[mU>(vTau1[None,:])]=2
        self.mG[0,iF0:iF1,iM0:iM1]=mG
    
    def __draw_y(self):
        if self.iChunks>1: tCount=tqdm(total=self.iChunks*self.iS)
        mEY=self.rng.normal(size=(self.iS,self.iF))
        if self.iS>1:
            mEY=self.mWeightSibE@mEY
        self.vGN=(self.dPropGN**0.5)\
            *((self.vGN-self.vGN.mean())/self.vGN.std())
        self.mEY=((1-(self.dHsqY+self.dPropGN))**0.5)\
            *((mEY-mEY.mean())/mEY.std())
        self.mGNnew=np.zeros((self.iS,self.iF))
        mGY=np.zeros((self.iS,self.iF))
        for h in range(self.iS):
            for j in range(self.iMT):
                iM0=self.iSM*j
                iM1=min(self.iM,iM0+self.iSM)
                vBeta=self.vBeta[iM0:iM1]
                vGamma=self.vGamma[iM0:iM1]
                for i in range(self.iFT):
                    iF0=self.iSFT*i
                    iF1=min(self.iF,iF0+self.iSFT)
                    mG=self.mG[h,iF0:iF1,iM0:iM1]
                    self.mGNnew[h,iF0:iF1]+=(mG*vGamma[None,:]).sum(axis=1)
                    mGY[h,iF0:iF1]+=(mG*vBeta[None,:]).sum(axis=1)
                    if self.iChunks>1: tCount.update(1)
        if self.iChunks>1: tCount.close()
        self.mGY=(self.dHsqY**0.5)*((mGY-mGY.mean())/mGY.std())
        self.mY=self.mGY+self.mEY+self.vGN[None,:]
        self.mAM=self.dCorrYAM*self.mY+\
            ((1-(self.dCorrYAM**2))**0.5)*self.rng.normal(size=(self.iS,\
                                                                self.iF))
    
    def __match(self):
        self.vGN=np.empty(self.iP)
        self.vMID=np.empty(self.iP,dtype=np.uint32)
        self.vPID=np.empty(self.iP,dtype=np.uint32)
        self.mGM=np.empty((self.iP,self.iM),dtype=np.uint8)
        self.mGP=np.empty((self.iP,self.iM),dtype=np.uint8)
        for i in range(self.iS):
            vIndM=((self.mFAM[i,:,4]==gnames.iFemale).nonzero())[0]
            vIndP=((self.mFAM[i,:,4]==gnames.iMale).nonzero())[0]
            vX1=self.rng.normal(size=self.iPS)
            vX2=self.dRhoAM*vX1+\
                ((1-(self.dRhoAM**2))**0.5)*self.rng.normal(size=self.iPS)
            vX1rank=vX1.argsort().argsort()
            vX2rank=vX2.argsort().argsort()
            vIndM=vIndM[self.mAM[i,vIndM].argsort()][vX1rank]
            vIndP=vIndP[self.mAM[i,vIndP].argsort()][vX2rank]
            self.vGN[i*self.iPS:(i+1)*self.iPS]=\
                self.mGNnew[i,vIndM]+self.mGNnew[i,vIndP]
            self.vMID[i*self.iPS:(i+1)*self.iPS]=self.mFAM[i,vIndM,1]
            self.vPID[i*self.iPS:(i+1)*self.iPS]=self.mFAM[i,vIndP,1]
            self.mGM[i*self.iPS:(i+1)*self.iPS]=self.mG[i,vIndM]
            self.mGP[i*self.iPS:(i+1)*self.iPS]=self.mG[i,vIndP]
        self.mG=None
    
    def __mate(self):
        self.mG=np.empty((self.iS,self.iF,self.iM),dtype=np.uint8)
        mG02=(self.mGM==2).astype(np.uint8)+(self.mGP==2).astype(np.uint8)
        if self.iChunks>1: tCount=tqdm(total=self.iChunks*self.iS)
        for h in range(self.iS):
            mGC=mG02.copy()
            for j in range(self.iMT):
                iM0=self.iSM*j
                iM1=min(self.iM,iM0+self.iSM)
                iM=iM1-iM0
                for i in range(self.iFT):
                    iF0=self.iSFT*i
                    iF1=min(self.iF,iF0+self.iSFT)
                    iF=iF1-iF0
                    mG1=np.zeros((iF,iM),dtype=np.uint8)
                    mM1=self.mGM[iF0:iF1,iM0:iM1]==1
                    mP1=self.mGP[iF0:iF1,iM0:iM1]==1
                    mG1[mM1]+=(self.rng.uniform(size=(mM1.sum()))>0.5)
                    mG1[mP1]+=(self.rng.uniform(size=(mP1.sum()))>0.5)
                    mGC[iF0:iF1,iM0:iM1]+=mG1
                    if self.iChunks>1: tCount.update(1)
            self.mG[h,:,:]=mGC
        if self.iChunks>1: tCount.close()
        mG02=None
        mGC=None
        self.mGM=None
        self.mGP=None
    
    def __write_fam(self,sName):
        mFAM=self.mFAM.reshape((self.iN,self.mFAM.shape[2]))
        vMiss=gnames.iMissY*np.ones((self.iN,1),dtype=np.int8)
        mFAMY=np.hstack((mFAM,vMiss))
        np.savetxt(sName+gnames.sFamExt,mFAMY,fmt='%i\t%i\t%i\t%i\t%i\t%i')
    
    def __write_phe(self,sName):
        mFIDIID=self.mFAM[:,:,0:2].reshape((self.iN,2))
        vY=self.mY.reshape((self.iN,1))
        mFIDIIDY=np.hstack((mFIDIID,vY))
        np.savetxt(sName+gnames.sPheExt,mFIDIIDY,fmt='%i\t%i\t%f')
    
    def __write_bim(self,sName):
        with open(sName+gnames.sBimExt,'w') as oFile:
            for j in range(self.iM):
                sSNP=str(self.vChr[j])+'\t'+self.lSNPs[j]+'\t'\
                    +str(self.vCM[j])+'\t'+str(self.vPos[j])+'\t'\
                        +self.vA1[j]+'\t'+self.vA2[j]+'\n'
                oFile.write(sSNP)
    
    def __write_bed(self,sName):
        iB=int(self.iN/gnames.iNperByte)
        iR=self.iN%gnames.iNperByte
        iBT=iB+(iR>0)
        mG=np.empty((iBT*gnames.iNperByte,self.iM),dtype=np.uint8)
        mG[0:((iB*gnames.iNperByte)+iR)]=2*self.mG.reshape((self.iN,self.iM))
        mG[mG==4]=3
        mG[((iB*gnames.iNperByte)+iR):iBT*gnames.iNperByte]=0
        vBase=np.array([2**0,2**2,2**4,2**6]*iBT,dtype=np.uint8)
        mBytes=(mG*vBase[:,None]).reshape(iBT,gnames.iNperByte,self.iM)\
            .sum(axis=1).astype(np.uint8)
        vBytes=mBytes.T.ravel()
        with open(sName+gnames.sBedExt,'wb') as oFile:
            oFile.write(gnames.binBED1)
            oFile.write(gnames.binBED2)
            oFile.write(gnames.binBED3)
            oFile.write(bytes(vBytes))
    
    def MakeBed(self,sName='genotypes'):
        """
        Export genotypes and phenotypes to PLINK files
        
        Attributes
        ----------
        sName : string, optional
            prefix for PLINK binary files to generate; default='genotypes'
        """
        if not(isinstance(sName,str)):
            raise ValueError('Prefix for PLINK binary files not a string')
        if sName=='':
            raise ValueError('Prefix for PLINK binary files is empty string')
        self.__write_phe(sName)
        self.__write_fam(sName)
        self.__write_bim(sName)
        self.__write_bed(sName)
    
    def PerformGWAS(self,sName='results'):
        """
        Perform classical GWAS and within-family GWAS based on offspring data
        
        Attributes
        ----------
        sName : string, optional
            prefix for GWAS files; default='results'
        """
        if self.iS==1:
            raise SyntaxError('Within-family GWAS not possible for 1 child '+\
                              'per family')
        if not(isinstance(sName,str)):
            raise ValueError('Prefix for GWAS files not a string')
        if sName=='':
            raise ValueError('Prefix for GWAS files is empty string')
        self.__do_standard_gwas(sName)
        self.__do_wf_gwas(sName)
    
    def __do_standard_gwas(self,sName=None,iC=None,vFamInd=None,bExport=True):
        if iC is None:
            iC=slice(None)
        else:
            iC=slice(0,iC+1,1)
        if vFamInd is None:
            vFamInd=slice(None)
        mY=self.mY[iC,vFamInd]-self.mY[iC,vFamInd].mean()
        iN=int(np.prod(mY.shape))
        vAF=np.zeros(self.iM)
        vXTY=np.zeros(self.iM)
        vXTX=np.zeros(self.iM)
        vB=np.zeros(self.iM)
        if bExport:
            if sName is None:
                raise ValueError('Prefix for GWAS files is not defined')
            vSSR=np.zeros(self.iM)
        for j in range(self.iMT):
            iM0=self.iSM*j
            iM1=min(self.iM,iM0+self.iSM)
            mG=self.mG[iC,vFamInd,iM0:iM1]
            vThisAF=mG.mean(axis=(0,1))/2
            vThisXTY=(mG*mY[:,:,None]).sum(axis=(0,1))
            vThisXTX=(mG**2).sum(axis=(0,1))-iN*((vThisAF*2)**2)
            vThisXTX[vThisXTX<np.finfo(float).eps]=np.nan
            vThisB=vThisXTY/vThisXTX
            vAF[iM0:iM1]=vThisAF
            vXTY[iM0:iM1]=vThisXTY
            vXTX[iM0:iM1]=vThisXTX
            vB[iM0:iM1]=vThisB
            if bExport:
                mYhat=mG*vThisB[None,None,:]
                vSSR[iM0:iM1]=(mY**2).sum()-2*((mYhat*mY[:,:,None])\
                                               .sum(axis=(0,1)))+\
                    (mYhat**2).sum(axis=(0,1))-iN*((mYhat.mean(axis=(0,1)))**2)
        if bExport:
            vSE=((vSSR/(iN-1))/vXTX)**0.5
            vT=vB/vSE
            vP=2*t.cdf(-abs(vT),iN-1)
            dfGWAS=pd.DataFrame((self.vA1,self.vA2,vB,vSE,vT,vP,vAF),\
                                columns=self.lSNPs,index=gnames.lGWAScol).T
            dfGWAS.index.name='SNP'
            dfGWAS.to_csv(sName+gnames.sGWASExt,sep='\t',na_rep='NA')
        else:
            return vB,vAF
    
    def __do_wf_gwas(self,sName=None,bExport=True):
        mY=self.mY-self.mY.mean(axis=0)[None,:]
        iC=mY.shape[0]
        iF=mY.shape[1]
        iN=int(iC*iF)
        vAF=np.zeros(self.iM)
        vXTY=np.zeros(self.iM)
        vXTX=np.zeros(self.iM)
        vB=np.zeros(self.iM)
        if bExport:
            if sName is None:
                raise ValueError('Prefix for GWAS files is not defined')
            vSSR=np.zeros(self.iM)
        for j in range(self.iMT):
            iM0=self.iSM*j
            iM1=min(self.iM,iM0+self.iSM)
            mG=self.mG[:,:,iM0:iM1]
            vAF[iM0:iM1]=(mG.mean(axis=(0,1)))/2
            vThisXTY=(mG*mY[:,:,None]).sum(axis=(0,1))
            vThisXTX=(((mG**2).sum(axis=0))-\
                      iC*((mG.mean(axis=0))**2)).sum(axis=0)
            vThisXTX[vThisXTX<np.finfo(float).eps]=np.nan
            vThisB=vThisXTY/vThisXTX
            vXTY[iM0:iM1]=vThisXTY
            vXTX[iM0:iM1]=vThisXTX
            vB[iM0:iM1]=vThisB
            if bExport:
                mYhat=mG*vThisB[None,None,:]
                vSSR[iM0:iM1]=(mY**2).sum()-\
                    2*((mYhat*mY[:,:,None]).sum(axis=(0,1)))+\
                        (mYhat**2).sum(axis=(0,1))-\
                            iC*(((mYhat.mean(axis=0))**2).sum(axis=0))
        if bExport:
            vSE=((vSSR/(iN-iF))/vXTX)**0.5
            vT=vB/vSE
            vP=2*t.cdf(-abs(vT),iN-1)
            dfGWAS_WF=pd.DataFrame((self.vA1,self.vA2,vB,vSE,vT,vP,vAF),\
                                columns=self.lSNPs,index=gnames.lGWAScol).T
            dfGWAS_WF.index.name='SNP'
            dfGWAS_WF.to_csv(sName+gnames.sWFExt,sep='\t',na_rep='NA')
        else:
            return vB,vAF
    
    def __compute_grm(self,dMAF,vFamInd):
        iF=len(vFamInd)
        iN=int(self.iS*iF)
        vEAF=(self.mG[:,vFamInd].mean(axis=(0,1)))/2
        vKeep=(vEAF>dMAF)*(vEAF<(1-dMAF))
        mA=np.zeros((iN,iN),dtype=np.float32)
        iM=vKeep.sum()
        iChunks=int(self.iMT*self.iS*(self.iS+1)/2)
        if iChunks>1: tCount=tqdm(total=iChunks)
        for j in range(self.iMT):
            iM0=self.iSM*j
            iM1=min(self.iM,iM0+self.iSM)
            vThisKeep=vKeep[iM0:iM1]
            vThisEAF=vEAF[iM0:iM1][vThisKeep]
            vD=1/(2*iM*vThisEAF*(1-vThisEAF))
            mG=self.mG[:,vFamInd,iM0:iM1][:,:,vThisKeep]
            for h in range(self.iS):
                mG1=mG[h]
                mX1=mG1*vD[None,:]
                mA[h*iF:(h+1)*iF,h*iF:(h+1)*iF]+=mX1@mG1.T
                if iChunks>1: tCount.update(1)
                for i in range(h+1,self.iS):
                    mG2=mG[i]
                    mA12=mX1@mG2.T
                    mA[h*iF:(h+1)*iF,i*iF:(i+1)*iF]+=mA12
                    mA[i*iF:(i+1)*iF,h*iF:(h+1)*iF]+=mA12.T
                    if iChunks>1: tCount.update(1)
        if iChunks>1: tCount.close()
        mA=mA-(mA.mean(axis=0)[None,:])
        mA=mA-(mA.mean(axis=1)[:,None])
        return mA,iM
    
    @staticmethod
    def __write_grm(sName,mA,iM):
        iN=int(mA.shape[0])
        (vIndR,vIndC)=np.tril_indices(iN)
        vA=(mA[vIndR,vIndC]).astype(np.float32)
        vM=(np.ones(vA.shape)*iM).astype(np.float32)
        vA.tofile(sName+gnames.sGrmBinExt)
        vM.tofile(sName+gnames.sGrmBinNExt)
    
    def __write_ids(self,sName,vFamInd):
        iN=len(vFamInd)*self.iS
        mFIDIID=self.mFAM[:,vFamInd,0:2].reshape((iN,2))
        np.savetxt(sName+gnames.sGrmIdExt,mFIDIID,fmt='%i\t%i')
    
    def MakeGRM(self,sName='genotypes',dMAF=0.01,vFamInd=None):
        """
        Make GRM in GCTA binary format
        
        Attributes
        ----------
        sName : string, optional
            prefix for binary GRM files; default='genotypes'
        
        dMAF : float in (0,0.45), optional
            SNPs with an empirical minor-allele frequency below this threshold
            are excluded from calculation of the GRM
        
        vFamInd : np.array, optional
            indices of families for which to construct GRM;
            default=None, which corresponds to all families
        """
        if not(isinstance(sName,str)):
            raise ValueError('Prefix for binary GRM files not a string')
        if sName=='':
            raise ValueError('Prefix for binary GRM files is empty string')
        if not(isinstance(dMAF,(int,float))):
            raise ValueError('Minor-allele-frequency threshold not a number')
        if dMAF<0:
            raise ValueError('Minor-allele-frequency threshold is negative')
        if dMAF>=gnames.dTooHighMAFThreshold:
            raise ValueError('Minor-allele-frequency threshold is'+\
                             ' unreasonably high')
        if vFamInd is None:
            vFamInd=np.arange(self.iF)
        (mA,iM)=self.__compute_grm(dMAF,vFamInd)
        gnames.__write_grm(sName,mA,iM)
        self.__write_ids(sName,vFamInd)
    
    def ComputeDiagsGRM(self,dMAF=0.01):
        """
        Compute diagonal elements of the GRM for the current generation
        
        Attributes
        ----------
        dMAF : float in (0,0.45), optional
            SNPs with an empirical minor-allele frequency below this threshold
            are excluded from calculation of the diagonal of the GRM
        """
        if not(isinstance(dMAF,(int,float))):
            raise ValueError('Minor-allele-frequency threshold not a number')
        if dMAF<0:
            raise ValueError('Minor-allele-frequency threshold is negative')
        if dMAF>=gnames.dTooHighMAFThreshold:
            raise ValueError('Minor-allele-frequency threshold is'+\
                             ' unreasonably high')
        vEAF=(self.mG.mean(axis=(0,1)))/2
        vKeep=(vEAF>dMAF)*(vEAF<(1-dMAF))
        vDiag=np.zeros(self.iN)
        iM=vKeep.sum()
        for j in range(self.iMT):
            iM0=self.iSM*j
            iM1=min(self.iM,iM0+self.iSM)
            vThisKeep=vKeep[iM0:iM1]
            vThisEAF=vEAF[iM0:iM1][vThisKeep]
            vM=2*vThisEAF
            vD=1/((iM*2*vThisEAF*(1-vThisEAF))**0.5)
            mG=self.mG[:,:,iM0:iM1][:,:,vThisKeep]
            for h in range(self.iS):
                mX=(mG[h]-vM[None,:])*vD[None,:]
                vDiag[h*self.iF:(h+1)*self.iF]+=(mX**2).sum(axis=1)
        return vDiag
        
    def __write_pgi_pheno_grm(self,sName,iFGWAS,iFPGI,dMAFThreshold):
        vInd=self.rng.permutation(self.iF)
        vFamInd1=np.sort(vInd[0:iFGWAS])
        vFamInd2=np.sort(vInd[iFGWAS:2*iFGWAS])
        vFamIndP=np.sort(vInd[0:2*iFGWAS])
        vFamIndOut=np.sort(vInd[2*iFGWAS:2*iFGWAS+iFPGI])
        vB1,vAF1=self.__do_standard_gwas(iC=0,vFamInd=vFamInd1,bExport=False)
        vB2,vAF2=self.__do_standard_gwas(iC=0,vFamInd=vFamInd2,bExport=False)
        vBP,vAFP=self.__do_standard_gwas(iC=0,vFamInd=vFamIndP,bExport=False)
        vDrop=(vAF1<=dMAFThreshold)|(vAF2<=dMAFThreshold)|\
            (vAFP<=dMAFThreshold)|(vAF1>=(1-dMAFThreshold))|\
                (vAF2>=(1-dMAFThreshold))|(vAFP>=(1-dMAFThreshold))
        vB1[vDrop]=0
        vB2[vDrop]=0
        vBP[vDrop]=0
        tShape=(self.iS*iFPGI,1)
        mPGI1=np.zeros((self.iS,iFPGI))
        mPGI2=np.zeros((self.iS,iFPGI))
        mPGIP=np.zeros((self.iS,iFPGI))
        vAF=np.zeros(self.iM)
        for j in range(self.iMT):
            iM0=self.iSM*j
            iM1=min(self.iM,iM0+self.iSM)
            mG=self.mG[:,vFamIndOut,iM0:iM1]
            mPGI1+=(mG*vB1[None,None,iM0:iM1]).sum(axis=2)
            mPGI2+=(mG*vB2[None,None,iM0:iM1]).sum(axis=2)
            mPGIP+=(mG*vBP[None,None,iM0:iM1]).sum(axis=2)
            for i in range(self.iFT):
                iF0=self.iSFT*i
                iF1=min(self.iF,iF0+self.iSFT)
                vAF[iM0:iM1]+=(self.mG[:,iF0:iF1,iM0:iM1].sum(axis=(0,1)))\
                    /(2*self.iN)
        vPGI1=mPGI1.reshape(tShape)
        vPGI2=mPGI2.reshape(tShape)
        vPGIP=mPGIP.reshape(tShape)
        vY=self.mY[:,vFamIndOut].reshape(tShape)
        vGY=self.mGY[:,vFamIndOut].reshape(tShape)
        vEY=self.mEY[:,vFamIndOut].reshape(tShape)
        vGN=np.tile(self.vGN[vFamIndOut],(self.iS,1)).reshape(tShape)
        vPGIT=vGY+vGN
        vAM=self.mAM[:,vFamIndOut].reshape(tShape)
        vFID=self.mFAM[:,vFamIndOut,0].reshape(tShape)
        vIID=self.mFAM[:,vFamIndOut,1].reshape(tShape)
        vPID=self.mFAM[:,vFamIndOut,2].reshape(tShape)
        vMID=self.mFAM[:,vFamIndOut,3].reshape(tShape)
        mData=np.hstack((vFID,vIID,vPID,vMID,vY,vGY,vEY,vGN,vAM,vPGIT,vPGI1,vPGI2,vPGIP))
        mY=np.hstack((vFID,vIID,vY))
        vMean=mData[:,4:].mean(axis=0)
        vStd=mData[:,4:].std(axis=0)
        vStd[vStd<np.finfo(float).eps]=1
        mData[:,4:]=(mData[:,4:]-vMean[None,:])/vStd[None,:]
        np.savetxt(sName+gnames.sPGIExt,mData,header=gnames.sPGIheader,\
                   fmt='%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f',\
                       comments='')
        np.savetxt(sName+gnames.sPheExt,mY,header='FID\tIID\tY',\
                   fmt='%i\t%i\t%f',comments='')
        iYMEFF=self.iM-((vAF<np.finfo(float).eps).sum()\
                        +(abs(vAF-1)<np.finfo(float).eps).sum())
        iPGIMEFF=self.iM-(vDrop.sum())
        with open(sName+gnames.sINFOExt,'w') as oInfoWriter:
            oInfoWriter.write('#SNPs directly affecting Y = '+str(iYMEFF)+'\n')
            oInfoWriter.write('#SNPs used to construct PGIs = '+str(iPGIMEFF))
        self.MakeGRM(sName,dMAFThreshold,vFamIndOut)
    
    def MakeThreePGIs(self,sName='results',iNGWAS=None,iNPGI=None,\
                      dMAFThreshold=0):
        """
        Perform 2 GWASs on non-overlapping samples, considering 1 child per
        family. Also perform a GWAS on these 2 GWAS samples pooled. Use these
        3 sets of GWAS estimates to construct 3 PGIs in the hold-out sample.
        For the hold-out sample, export these PGIs, the GRM, and phenotype.
                
        Attributes
        ----------
        sName : string, optional
            prefix for binary GRM files; default='results'
        
        iNGWAS : int, optional
            sample size of the two non-overlapping discovery GWASs;
            default=None, which corresponds to using 40% of the families
            for each GWAS and 80% for the pooled GWAS, drawing 1 sibling
            per included family
        
        iNPGI : int, optional
            sample size for calculating PGIs;
            default=None, which corresponds to using 20% of the families,
            drawing all siblings per included family
        
        dMAFThreshold : float in [0,0.45), optional
            MAF threshold that SNPs need to meet in all GWAS samples to be
            included in PGIs; default=0, which corresponds to excluding SNPs
            with MAF exactly equal to zero
        """
        if iNGWAS is None:
            iNGWAS=int(gnames.dPropGWAS*self.iF)
        if iNPGI is None:
            iNPGI=int(gnames.dPropPGI*self.iF)*self.iS
        if not(isinstance(iNGWAS,int)):
            raise ValueError('GWAS sample size non-integer')
        if not(isinstance(iNPGI,int)):
            raise ValueError('PGI sample size non-integer')
        if iNPGI%self.iS>0:
            raise ValueError('PGI sample size not divisible by '+\
            'number of siblings in this generation')
        if not(isinstance(dMAFThreshold,(int,float))):
            raise ValueError('Minor-allele-frequency threshold not a number')
        if iNGWAS<1:
            raise ValueError('GWAS sample size non-positive')
        if iNPGI<1:
            raise ValueError('PGI sample size non-positive')
        if ((2*self.iS*iNGWAS)+iNPGI)>(self.iN):
            raise ValueError('N too low for desired N(GWAS) and N(PGI)')
        if dMAFThreshold<0:
            raise ValueError('Minor-allele-frequency threshold is negative')
        if dMAFThreshold>=gnames.dTooHighMAFThreshold:
            raise ValueError('Minor-allele-frequency threshold is '+\
                             'unreasonably high')
        iFGWAS=iNGWAS
        iFPGI=int(iNPGI/self.iS)
        self.__write_pgi_pheno_grm(sName,iFGWAS,iFPGI,dMAFThreshold)
    
    def Test():
        """
        Function to test if gnames works properly
        """
        dTimeStart=time.time()
        iN=1000
        iM=10000
        iT=10
        print('TEST OF GNAMES')
        print('with 1000 founders, 10,000 SNPs, and two children per pair')
        print('INITIALISING SIMULATOR')
        simulator=gnames(iN,iM)
        print('Highest diagonal element of GRM for founders = '+\
              str(round(max(simulator.ComputeDiagsGRM()),3)))
        print('SIMULATING '+str(iT)+' GENERATIONS')
        simulator.Simulate(iT)
        print('Highest diagonal element of GRM after '+str(iT)+\
              ' generations = '+str(round(max(simulator.ComputeDiagsGRM()),3)))
        dTime=time.time()-dTimeStart
        print('GENERATING OUTPUT')
        print('Calculating and storing classical GWAS and within-family GWAS')
        print('results based on offspring data last generation')
        simulator.PerformGWAS()
        print('Writing PLINK files (genotypes.bed,.bim,.fam,.phe)')
        simulator.MakeBed()
        print('Making GRM in GCTA binary format')
        print('(genotypes.grm.bin,.grm.N.bin,.grm.id)')
        simulator.MakeGRM()
        print('Making GRM and 3 PGIs in hold-out sample based on 3 sets of')
        print('GWAS estimates (GWAS 1 & 2: non-overlapping; GWAS 3: pooled;')
        print('all sampling 1 child per family)')
        simulator.MakeThreePGIs(dMAFThreshold=0.01)
        print('Runtime: '+str(round(dTime,3))+' seconds')
