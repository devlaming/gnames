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
        number of founders
    
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
    
    iSN : int >= 0, optional
        block size of founders when generating founder genotypes;
        default=0, which is treated as having one big block for all founders
    
    iSM : int >= 0, optional
        block size of SNPs when generating founder genotypes;
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
    
    MakeGRM(sName='genotypes',dMAF=0.01)
        Make GRM in GCTA binary format
    
    MakeBed(sName='genotypes')
        Export genotypes and phenotypes to PLINK files
    
    PerformGWAS(sName='results')
        Perform classical GWAS and within-family GWAS based on offspring data
    
    MakeTwoPGIs(sName='results',iNGWAS=None,iNPGI=None)
        Make 2 PGIs in hold-out sample based on 2 sets of GWAS estimates for Y
        (GWASs use non-overlapping samples and data on only child per family)
    '''
    dTooHighMAFThreshold=0.45
    tIDs=('FID','IID')
    sSNPIDs='SNP'
    lPheno=['Y']
    sMat='Mother'
    sPat='Father'
    sBedExt='.bed'
    sBimExt='.bim'
    sFamExt='.fam'
    sPheExt='.phe'
    sMissY='-9'
    sGrmBinExt='.grm.bin'
    sGrmBinNExt='.grm.N.bin'
    sGrmIdExt='.grm.id'
    sGWASExt='.GWAS.classical.txt'
    sWFExt='.GWAS.within_family.txt'
    sPGIExt='.pgs'
    binBED1=bytes([0b01101100])
    binBED2=bytes([0b00011011])
    binBED3=bytes([0b00000001])
    iNperByte=4
    lAlleles=['A','C','G','T']
    iPloidy=2
    lGWAScol=['Baseline Allele','Effect Allele','Per-allele effect estimate',\
               'Standard error','T-test statistic','P-value']
    dPropGWAS=0.4
    dPropPGI=0.2
    def __init__(self,iN,iM,iC=2,dHsqY=0.5,dPropGN=0.25,dCorrYAM=1,\
                 dRhoAM=0.5,dRhoSibE=0,iSN=0,iSM=0,\
                 dBetaAF0=0.35,dMAF0=0.1,iSeed=502421368):
        if not(isinstance(iN,int)):
            raise ValueError('Sample size not integer')
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
        if not(isinstance(iSN,int)):
            raise ValueError('Block size for indiviuals not integer')
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
            raise ValueError('Sample size less than two')
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
        if iSN<0:
            raise ValueError('Block size for individuals negative')
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
        self.iN=iN
        self.iM=iM
        self.iC=iC
        self.dHsqY=dHsqY
        self.dPropGN=dPropGN       
        self.dCorrYAM=dCorrYAM       
        self.dRhoAM=dRhoAM
        self.dRhoSibE=dRhoSibE
        self.iSN=iSN
        self.iSM=iSM
        self.rng=np.random.RandomState(iSeed)
        self.__set_loadings_rho_sib_e()
        self.__draw_alleles()
        self.__draw_afs(dBetaAF0,dMAF0)
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
        print('Drawing alleles for SNPs of founders')
        self.vChr=np.zeros(self.iM,dtype=np.uint8)
        self.lSNPs=['SNP_'+str(i+1) for i in range(self.iM)]
        lA1A2=[self.rng.choice(gnames.lAlleles,size=gnames.iPloidy,\
                               replace=False) for i in range(self.iM)]
        mA1A2=np.array(lA1A2)
        self.vA1=mA1A2[:,0]
        self.vA2=mA1A2[:,1]
        self.vCM=np.zeros(self.iM,dtype=np.uint8)
        self.vPos=np.arange(self.iM)+1
    
    def __draw_afs(self,dBetaAF0,dMAF0):
        print('Drawing allele frequencies for SNPs of founders')
        vAF=self.rng.beta(dBetaAF0,dBetaAF0,self.iM)
        while (min(vAF)<dMAF0)|(max(vAF)>(1-dMAF0)):
            vAF[vAF<dMAF0]=self.rng.beta(dBetaAF0,dBetaAF0,np.sum(vAF<dMAF0))
            vAF[vAF>(1-dMAF0)]=\
                self.rng.beta(dBetaAF0,dBetaAF0,np.sum(vAF>(1-dMAF0)))
        self.vAF0=vAF
        self.vTau0=(1-vAF)**2
        self.vTau1=1-(vAF**2)
    
    def __draw_betas(self):
        print('Drawing true SNP effects')
        vScaling=(2*self.vAF0*(1-self.vAF0))**(-0.5)
        self.vBetaHsq=self.rng.normal(size=self.iM)*vScaling
        self.vBetaGN=self.rng.normal(size=self.iM)*vScaling
    
    def __draw_gen0(self):
        self.__draw_g0()
        self.__draw_y()
    
    def __draw_next_gen(self):
        self.__match()
        self.__mate()
        self.__draw_y()
    
    def __draw_g0(self):
        print('Drawing genotypes founders')
        self.iT=0
        if self.iSN==0:
            iSN=self.iN
        else:
            iSN=self.iSN
        iB=int(self.iN/iSN)
        iR=(self.iN)%(iSN)
        iBT=iB+(iR>0)
        self.mG=np.empty((1,self.iN,self.iM),dtype=np.int8)
        if iBT>1: tCount=tqdm(total=iBT)
        for i in range(iB):
            self.__draw_g0_rows(iSN*i,iSN)
            if iBT>1: tCount.update(1)
        if iR>0:
            self.__draw_g0_rows(iSN*iB,iR)
            if iBT>1: tCount.update(1)
        if iBT>1: tCount.close()
    
    def __draw_g0_rows(self,iNstart,iNadd):
        if self.iSM==0:
            iSM=self.iM
        else:
            iSM=self.iSM
        iB=int(self.iM/iSM)
        iR=(self.iM)%(iSM)
        for i in range(iB):
            self.__draw_g0_rows_cols(iNstart,iNadd,iSM*i,iSM)
        if iR>0:
            self.__draw_g0_rows_cols(iNstart,iNadd,iSM*iB,iR)
    
    def __draw_g0_rows_cols(self,iNstart,iNadd,iMstart,iMadd):
        mU=self.rng.uniform(size=(iNadd,iMadd))
        mThisG=np.ones((iNadd,iMadd),dtype=np.int8)
        mThisG[mU<(self.vTau0[None,iMstart:iMstart+iMadd])]=0
        mThisG[mU>(self.vTau1[None,iMstart:iMstart+iMadd])]=2
        self.mG[0,iNstart:iNstart+iNadd,iMstart:iMstart+iMadd]=mThisG
    
    def __mate(self):
        self.iT+=1
        self.iN=self.mGM.shape[0]
        self.mG=np.empty((self.iC,self.iN,self.iM),dtype=np.int8)
        mCM=np.zeros((self.iN,self.iM),dtype=np.int8)
        mCF=np.zeros((self.iN,self.iM),dtype=np.int8)
        mCM[self.mGM==2]=1
        mCF[self.mGF==2]=1
        for i in range(self.iC):
            mGC=mCM+mCF
            mGC[self.mGM==1]+=\
                (self.rng.uniform(size=((self.mGM==1).sum()))>0.5)
            mGC[self.mGF==1]+=\
                (self.rng.uniform(size=((self.mGF==1).sum()))>0.5)
            self.mG[i,:,:]=mGC
    
    def __draw_y(self):
        iC=1
        if self.iT>0:
            iC=self.iC
        self.mGN=(self.mG*self.vBetaGN[None,None,:]).mean(axis=2)
        if self.iT>0:
            vGNold=(self.dPropGN**0.5)*((self.vGNold-self.vGNold.mean())\
                                        /self.vGNold.std())
        else:
            vGNold=(self.dPropGN**0.5)*self.rng.normal(size=self.iN)
        mGY=(self.mG*self.vBetaHsq[None,None,:]).mean(axis=2)
        mEY=self.rng.normal(size=(iC,self.iN))
        if self.iT>0:
            mEY=self.mWeightSibE@mEY
        mGY=(self.dHsqY**0.5)*((mGY-mGY.mean())/mGY.std())
        mEY=((1-(self.dHsqY+self.dPropGN))**0.5)*((mEY-mEY.mean())/mEY.std())
        self.mY=mGY+mEY+vGNold[None,:]
        self.mAM=self.dCorrYAM*self.mY\
            +((1-(self.dCorrYAM**2))**0.5)*self.rng.normal(size=(iC,self.iN))
    
    def __match(self):
        iC=1
        if self.iT>0:
            iC=self.iC
        self.vGNold=np.empty((int(iC*self.iN/2)))
        self.vYM=np.empty((int(iC*self.iN/2)))
        self.vYF=np.empty((int(iC*self.iN/2)))
        self.mGM=np.empty((int(iC*self.iN/2),self.iM),dtype=np.int8)
        self.mGF=np.empty((int(iC*self.iN/2),self.iM),dtype=np.int8)
        for i in range(iC):
            vInd=self.rng.permutation(self.iN)
            vIndM=vInd[0:int(self.iN/2)]
            vIndF=vInd[int(self.iN/2):]
            vX1=self.rng.normal(size=int(self.iN/2))
            vX2=self.dRhoAM*vX1+\
                ((1-self.dRhoAM**2)**0.5)*self.rng.normal(size=int(self.iN/2))
            vX1rank=vX1.argsort().argsort()
            vX2rank=vX2.argsort().argsort()
            vIndM=vIndM[self.mAM[i,vIndM].argsort()][vX1rank]
            vIndF=vIndF[self.mAM[i,vIndF].argsort()][vX2rank]
            self.vGNold[i*int(self.iN/2):(i+1)*int(self.iN/2)]=\
                self.mGN[i,vIndM]+self.mGN[i,vIndF]
            self.vYM[i*int(self.iN/2):(i+1)*int(self.iN/2)]=\
                self.mY[i,vIndM]
            self.vYF[i*int(self.iN/2):(i+1)*int(self.iN/2)]=\
                self.mY[i,vIndF]
            self.mGM[i*int(self.iN/2):(i+1)*int(self.iN/2)]=self.mG[i,vIndM]
            self.mGF[i*int(self.iN/2):(i+1)*int(self.iN/2)]=self.mG[i,vIndF]
    
    def __assign_ids(self):
        if self.iT<1:
            raise SyntaxError('Cannot assign IDs for founders')
        iF=self.mGM.shape[0]
        self.lFID=['Generation'+str(self.iT)+'_Family'+str(i+1)\
              for i in range(iF)]
        self.lIM=[gnames.sMat]*iF
        self.lIF=[gnames.sPat]*iF
        self.lIC=['']*self.iC
        for i in range(self.iC):
            self.lIC[i]=['Child'+str(i+1)]*iF
    
    def __create_dataframes(self,bPhenoOnly=False):
        if self.iT<1:
            raise SyntaxError('Cannot create DataFrames for founders')
        self.__assign_ids()
        miM=pd.MultiIndex.from_arrays([self.lFID,self.lIM],names=gnames.tIDs)
        miF=pd.MultiIndex.from_arrays([self.lFID,self.lIF],names=gnames.tIDs)
        dfY=pd.DataFrame(self.vYM,miM,gnames.lPheno)        
        dfY=dfY.append(pd.DataFrame(self.vYF,miF,gnames.lPheno))
        if not(bPhenoOnly):
            dfG=pd.DataFrame(self.mGM,miM,self.lSNPs)
            dfG=dfG.append(pd.DataFrame(self.mGF,miF,self.lSNPs))
        for i in range(self.iC):
            miC=pd.MultiIndex.from_arrays([self.lFID,self.lIC[i]],\
                                          names=gnames.tIDs)
            dfY=dfY.append(pd.DataFrame(self.mY[i],miC,gnames.lPheno))
            if not(bPhenoOnly):
                dfG=dfG.append(pd.DataFrame(self.mG[i],miC,self.lSNPs))
        self.dfY=dfY    
        if not(bPhenoOnly):
            self.dfG=dfG
    
    def __write_fam(self,sName):
        with open(sName+gnames.sFamExt,'w') as oFile:
            for j in range(len(self.dfG)):
                sFID=self.dfG.index[j][0]
                sIID=self.dfG.index[j][1]
                if sIID!=gnames.sMat and sIID!=gnames.sPat:
                    sMID=gnames.sMat
                    sPID=gnames.sPat
                else:
                    sMID='0'
                    sPID='0'
                sIND=sFID+'\t'+sIID+'\t'+sPID+'\t'+sMID+'\t0\t'\
                    +gnames.sMissY+'\n'
                oFile.write(sIND)
    
    def __write_phe(self,sName):
        with open(sName+gnames.sPheExt,'w') as oFile:
            sHeader=gnames.tIDs[0]+'\t'+gnames.tIDs[1]+'\t'\
                +gnames.lPheno[0]+'\n'
            oFile.write(sHeader)
            for j in range(len(self.dfY)):
                sFID=self.dfY.index[j][0]
                sIID=self.dfY.index[j][1]
                dY=self.dfY.values[j][0]
                sIND=sFID+'\t'+sIID+'\t'+str(dY)+'\n'
                oFile.write(sIND)
    
    def __write_bim(self,sName):
        with open(sName+gnames.sBimExt,'w') as oFile:
            for j in range(self.iM):
                sSNP=str(self.vChr[j])+'\t'+self.lSNPs[j]+'\t'\
                    +str(self.vCM[j])+'\t'+str(self.vPos[j])+'\t'\
                        +self.vA1[j]+'\t'+self.vA2[j]+'\n'
                oFile.write(sSNP)
    
    def __write_bed(self,sName):
        iN=self.dfG.shape[0]
        iB=int(iN/gnames.iNperByte)
        iR=iN%gnames.iNperByte
        iBT=iB+(iR>0)
        mG=np.empty((iBT*gnames.iNperByte,self.iM),dtype=np.uint8)
        mG[0:((iB*gnames.iNperByte)+iR)]=2*self.dfG.values
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
        if self.iT<1:
            raise SyntaxError('Cannot export to PLINK binary format for '+\
                              'founders')
        if not(isinstance(sName,str)):
            raise ValueError('Prefix for PLINK binary files not a string')
        if sName=='':
            raise ValueError('Prefix for PLINK binary files is empty string')
        self.__create_dataframes()
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
        if self.iT<1:
            raise SyntaxError('Cannot perform GWAS for founders')
        if not(isinstance(sName,str)):
            raise ValueError('Prefix for GWAS files not a string')
        if sName=='':
            raise ValueError('Prefix for GWAS files is empty string')
        self.__do_standard_gwas(sName)
        self.__do_wf_gwas(sName)
    
    def __do_standard_gwas(self,sName=None,iC=None,iN0=None,iN1=None,\
                           bExport=True):
        if iC is None:
            iC=self.mY.shape[0]
        if iN0 is None:
            iN0=0
        if iN1 is None:
            iN1=self.mY.shape[1]
        mY=self.mY[0:iC,iN0:iN1]-self.mY[0:iC,iN0:iN1].mean()
        iN=int(np.prod(mY.shape))
        vXTY=(self.mG[0:iC,iN0:iN1]*mY[:,:,None]).sum(axis=(0,1))
        vXTX=(self.mG[0:iC,iN0:iN1]**2).sum(axis=(0,1))-\
            iN*((self.mG[0:iC,iN0:iN1].mean(axis=(0,1)))**2)
        vXTX[vXTX<np.finfo(float).eps]=np.nan
        vB=vXTY/vXTX
        if bExport:
            if sName is None:
                raise ValueError('Prefix for GWAS files is not defined')
            mYhat=self.mG[0:iC,iN0:iN1]*vB[None,None,:]
            vSSR=(mY**2).sum()-2*((mYhat*mY[:,:,None]).sum(axis=(0,1)))+\
                (mYhat**2).sum(axis=(0,1))-iN*((mYhat.mean(axis=(0,1)))**2)
            vSE=((vSSR/(iN-1))/vXTX)**0.5
            vT=vB/vSE
            vP=2*t.cdf(-abs(vT),iN-1)
            dfGWAS=pd.DataFrame((self.vA1,self.vA2,vB,vSE,vT,vP),\
                                columns=self.lSNPs,index=gnames.lGWAScol).T
            dfGWAS.index.name=gnames.sSNPIDs
            dfGWAS.to_csv(sName+gnames.sGWASExt,sep='\t',na_rep='NA')
        else:
            return vB
    
    def __do_wf_gwas(self,sName=None,bExport=True):
        mY=self.mY-self.mY.mean(axis=0)[None,:]
        iC=mY.shape[0]
        iF=mY.shape[1]
        iN=int(np.prod(mY.shape))
        vXTY=(self.mG*mY[:,:,None]).sum(axis=(0,1))
        vXTX=(((self.mG**2).sum(axis=0))-\
            iC*((self.mG.mean(axis=0))**2)).sum(axis=0)
        vXTX[vXTX<np.finfo(float).eps]=np.nan
        vB=vXTY/vXTX
        if bExport:
            if sName is None:
                raise ValueError('Prefix for GWAS files is not defined')
            mYhat=(self.mG*vB[None,None,:])
            vSSR=(mY**2).sum()-2*((mYhat*mY[:,:,None]).sum(axis=(0,1)))+\
                (mYhat**2).sum(axis=(0,1))-\
                    iC*(((mYhat.mean(axis=0))**2).sum(axis=0))
            vSE=((vSSR/(iN-iF))/vXTX)**0.5
            vT=vB/vSE
            vP=2*t.cdf(-abs(vT),iN-1)
            dfGWAS_WF=pd.DataFrame((self.vA1,self.vA2,vB,vSE,vT,vP),\
                                columns=self.lSNPs,index=gnames.lGWAScol).T
            dfGWAS_WF.index.name=gnames.sSNPIDs
            dfGWAS_WF.to_csv(sName+gnames.sWFExt,sep='\t',na_rep='NA')
        else:
            return vB
    
    def __compute_grm(self,dMAF):
        vEAF=self.mG.mean(axis=(0,1))/2
        vKeep=(vEAF>dMAF)*(vEAF<(1-dMAF))
        iM=vKeep.sum()
        vEAF=vEAF[vKeep]
        mX=(self.mG[:,:,vKeep]-2*vEAF[None,None,:])/\
            (((2*vEAF*(1-vEAF))**0.5)[None,None,:])
        iN=int(self.mG.shape[0]*self.mG.shape[1])
        mA=((np.tensordot(mX,mX,axes=(2,2)))/iM).reshape((iN,iN))\
            .astype(np.float32)
        return mA,iM
    
    def __write_grm(self,sName,mA,iM):
        iN=int(self.mG.shape[0]*self.mG.shape[1])
        (vIndR,vIndC)=np.tril_indices(iN)
        vA=(mA[vIndR,vIndC]).astype(np.float32)
        vM=(np.ones(vA.shape)*iM).astype(np.float32)
        vA.tofile(sName+gnames.sGrmBinExt)
        vM.tofile(sName+gnames.sGrmBinNExt)
    
    def __write_ids(self,sName):
        self.__assign_ids()
        with open(sName+gnames.sGrmIdExt,'w') as oFile:
            for i in range(self.iC):
                sFIDsIIDs=''
                for j in range(self.iN):
                    sFIDsIIDs+=self.lFID[j]+"\t"+self.lIC[i][j]+"\n"
                oFile.write(sFIDsIIDs)
    
    def MakeGRM(self,sName='genotypes',dMAF=0.01):
        """
        Make GRM in GCTA binary format
        
        Attributes
        ----------
        sName : string, optional
            prefix for binary GRM files; default='genotypes'
        
        dMAF : float in (0,0.45), optional
            SNPs with an empirical minor-allele frequency below this threshold
            are excluded from calculation of the GRM
        """
        if self.iT<1:
            raise SyntaxError('Cannot create GRM for founders')
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
        (mA,iM)=self.__compute_grm(dMAF)
        self.__write_grm(sName,mA,iM)
        self.__write_ids(sName)
    
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
        vEAF=self.mG.mean(axis=(0,1))/2
        vKeep=(vEAF>dMAF)*(vEAF<(1-dMAF))
        vEAF=vEAF[vKeep]
        vDiag=(((self.mG[:,:,vKeep]-2*vEAF[None,None,:])/\
                   (((2*vEAF*(1-vEAF))**0.5)[None,None,:]))**2)\
            .mean(axis=2).ravel()
        return vDiag
    
    def __pgi_to_dataframe(self,iGWASNo,mPGI,iN0,iN1):
        lFID=self.lFID[iN0:iN1]
        dfPGI=pd.DataFrame()
        for i in range(self.iC):
            lIC=self.lIC[i][iN0:iN1]
            miC=pd.MultiIndex.from_arrays([lFID,lIC],names=gnames.tIDs)
            dfPGI=dfPGI.append(pd.DataFrame(mPGI[i],miC,\
                                            ['PGI from GWAS'+str(iGWASNo)]))
        return dfPGI
    
    def __calculate_pgi(self,iN0GWAS,iN1GWAS,iN0PGI,iN1PGI):
        vB=self.__do_standard_gwas(iC=1,iN0=iN0GWAS,iN1=iN1GWAS,bExport=False)
        mPGI=(self.mG[:,iN0PGI:iN1PGI]*vB[None,None,:]).sum(axis=2)
        return mPGI
    
    def __write_pgs(self,sName,iNGWAS,iNPGI):
        mPGI1=self.__calculate_pgi(0,iNGWAS,2*iNGWAS,2*iNGWAS+iNPGI)
        mPGI2=self.__calculate_pgi(iNGWAS,2*iNGWAS,2*iNGWAS,2*iNGWAS+iNPGI)
        dfPGI1=self.__pgi_to_dataframe(1,mPGI1,2*iNGWAS,2*iNGWAS+iNPGI)
        dfPGI2=self.__pgi_to_dataframe(2,mPGI2,2*iNGWAS,2*iNGWAS+iNPGI)
        dfPGI=dfPGI1.join(dfPGI2)
        dfPGI.to_csv(sName+gnames.sPGIExt,sep='\t',na_rep='NA')
    
    def MakeTwoPGIs(self,sName='results',iNGWAS=None,iNPGI=None):
        """
        Make 2 PGIs in hold-out sample based on 2 sets of GWAS estimates for Y
        (GWASs use non-overlapping samples and data on only child per family)
                
        Attributes
        ----------
        sName : string, optional
            prefix for binary GRM files; default='results'
        
        iNGWAS : int, optional
            sample size of the two non-overlapping discovery GWASs;
            default=None, which corresponds to using 40% of the families
            for each GWAS
        
        iNPGI : int, optional
            sample size for calculating PGIs;
            default=None, which corresponds to using 20% of the families
        """
        if self.iT<1:
            raise SyntaxError('Cannot create PGIs for founders')
        if iNGWAS is None:
            iNGWAS=int(gnames.dPropGWAS*self.iN)
        if iNPGI is None:
            iNPGI=int(gnames.dPropPGI*self.iN)
        if not(isinstance(iNGWAS,int)):
            raise ValueError('GWAS sample size non-integer')
        if not(isinstance(iNPGI,int)):
            raise ValueError('PGI sample size non-integer')
        if iNGWAS<1:
            raise ValueError('GWAS sample size non-positive')
        if iNPGI<1:
            raise ValueError('PGI sample size non-positive')
        if (2*iNGWAS+iNPGI)>self.iN:
            raise ValueError('N too low for desired N(GWAS) and N(PGI)')
        self.__create_dataframes(bPhenoOnly=True)
        self.__write_phe(sName)
        self.__write_pgs(sName,iNGWAS,iNPGI)
    
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
        print('Making 2 PGIs in hold-out sample based on 2 sets of')
        print('GWAS estimates for Y (GWASs use non-overlapping samples')
        print('and data on only child per family)')
        simulator.MakeTwoPGIs()
        print('Runtime: '+str(round(dTime,3))+' seconds')
