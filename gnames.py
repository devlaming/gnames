import time
import pandas as pd
import numpy as np
from tqdm import tqdm

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
        heritability of main trait Y without genetic-nurture effects;
        default=0.5
    
    dHsqAM : float in [0,1], optional
        heritability of assortative-mating trait AM; default=0.5
    
    dRhoG : float in [-1,+1], optional
        genetic correlation of AM and Y without genetic-nurture effects;
        default=1
        
    dRhoE : float in [-1,+1], optional
        environment correlation of AM and Y without genetic-nurture effects;
        default=1
    
    dRhoAM : float in [-1,+1], optional
        assortattive-mating strenght = correlation in AM between mates;
        default=0.8
    
    dVarGN : float >= 0, optional
        variance in Y accounted for by genetic-nurture effects, where Y
        without genetic-nurture effects has variance one; default=1
    
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
        Compute diagonals of the GRM for the current generation, excluding
        SNPs with a minor allele frequency below the given threshold
    
    MakeBed(sName='genotypes')
        Export genotypes to PLINK binary file format
    
    PerformGWAS()
        Perform classical GWAS and within-family GWAS based on offspring data
    '''
    dTooHighMAF=0.45
    tIDs=('FID','IID')
    lPheno=['Y']
    sMat='Mother'
    sPat='Father'
    sBedExt='.bed'
    sBimExt='.bim'
    sFamExt='.fam'
    binBED1=bytes([0b01101100])
    binBED2=bytes([0b00011011])
    binBED3=bytes([0b00000001])
    iNperByte=4
    def __init__(self,iN,iM,iC=2,dHsqY=0.5,dHsqAM=0.5,dRhoG=1,dRhoE=1,\
                 dRhoAM=0.8,dVarGN=1,iSN=0,iSM=0,dBetaAF0=0.35,dMAF0=0.1,\
                     iSeed=502421368):
        if not(isinstance(iN,int)):
            raise ValueError('Sample size not integer')
        if not(isinstance(iM,int)):
            raise ValueError('Number of SNPs not integer')
        if not(isinstance(iC,int)):
            raise ValueError('Number of children not integer')
        if not(isinstance(iSN,int)):
            raise ValueError('Block size for indiviuals not integer')
        if not(isinstance(iSM,int)):
            raise ValueError('Block size for SNPs not integer')
        if not(isinstance(iSeed,int)):
            raise ValueError('Seed for random-number generator not integer')
        if not(isinstance(dBetaAF0,(int,float))):
            raise ValueError('Parameter of beta distribution used to draw'+\
                             ' allele frequencies not a number')
        if not(isinstance(dMAF0,(int,float))):
            raise ValueError('Minor-allele-frequency threshold not a number')
        if not(isinstance(dHsqY,(int,float))):
            raise ValueError('Heritability of main trait Y is not a number')
        if not(isinstance(dHsqAM,(int,float))):
            raise ValueError('Heritability of assortative-mating trait is'+\
                             ' not a number')
        if not(isinstance(dRhoG,(int,float))):
            raise ValueError('Genetic correlation Y and'+\
                             ' assortative-mating trait is not a number')
        if not(isinstance(dRhoE,(int,float))):
            raise ValueError('Environment correlation Y and'+\
                             ' assortative-mating trait is not a number')
        if not(isinstance(dRhoAM,(int,float))):
            raise ValueError('Degree of assortative mating is not a number')
        if not(isinstance(dVarGN,(int,float))):
            raise ValueError('Variance accounted for by genetic nurture is'+\
                             ' not a number')
        if iN<2:
            raise ValueError('Sample size less than two')
        if iM<1:
            raise ValueError('Number of SNPs less than one')
        if iC<1:
            raise ValueError('Number of children less than one')
        if iSN<0:
            raise ValueError('Block size for individuals negative')
        if iSM<0:
            raise ValueError('Block size for SNPs negative')
        if iSeed<0:
            raise ValueError('Seed for random-number generator negative')
        if dBetaAF0<=0:
            raise ValueError('Parameter for beta distribution to draw'+\
                             ' allele frequencies is non-positive')
        if dMAF0<=0:
            raise ValueError('Minor-allele-frequency threshold is'+\
                             ' non-positive')
        if dMAF0>=gnames.dTooHighMAF:
            raise ValueError('Minor-allele-frequency threshold is'+\
                             ' unreasonably high')
        if dHsqY>1 or dHsqY<0:
            raise ValueError('Heritability of main trait Y is'+\
                             ' not constrained to [0,1] interval')
        if dHsqAM>1 or dHsqAM<0:
            raise ValueError('Heritability of assortative-mating trait is'+\
                             ' not constrained to [0,1] interval')
        if dRhoG>1 or dRhoG<-1:
            raise ValueError('Genetic correlation Y and assortative-mating'+\
                             ' trait is not constrained to [-1,+1] interval')
        if dRhoE>1 or dRhoE<-1:
            raise ValueError('Environment correlation Y and assortative-'+\
                             'mating trait is not constrained to [-1,+1]'+\
                                 ' interval')
        if dRhoAM>1 or dRhoAM<-1:
            raise ValueError('Degree of assortative mating is not'+\
                             ' not constrained to [-1,+1] interval')
        if dVarGN<0:
            raise ValueError('Variance accounted for by genetic nurture is'+\
                             ' negative')
        self.iN=iN
        self.iM=iM
        self.iC=iC
        self.iSN=iSN
        self.iSM=iSM
        self.dHsqY=dHsqY
        self.dHsqAM=dHsqAM
        self.dRhoG=dRhoG
        self.dRhoE=dRhoE
        self.dRhoAM=dRhoAM
        self.dVarGN=dVarGN
        self.rng=np.random.RandomState(iSeed)
        self.__draw_afs(dBetaAF0,dMAF0)
        self.__draw_betas()
        self.__draw_gen0()
    
    def Simulate(self,iGenerations=1):
        """
        Simulate data for a given number of new generations
        
        Simulates offspring genotypes and phenotypes under assortative mating
        of parents, where the phenotypes include Y and AM, where Y is subject
        to genetic nurture effects, and AM is the assortative-mating trait
        
        Attributes
        ----------
        iGenerations : int > 0, optional
            number of new generations to simulate data for; default=1
        """
        if not(isinstance(iGenerations,int)):
            raise ValueError('Number of generations not integer')
        if iGenerations<1:
            raise ValueError('Number of generations non-positive')
        for i in range(iGenerations):
            self.__draw_next_gen()
    
    def __draw_afs(self,dBetaAF0,dMAF0):
        print('Drawing allele frequencies SNPs founders')
        vAF=self.rng.beta(dBetaAF0,dBetaAF0,self.iM)
        while (min(vAF) < dMAF0) | (max(vAF)>(1-dMAF0)):
            vAF[vAF<dMAF0]=self.rng.beta(dBetaAF0,dBetaAF0,np.sum(vAF<dMAF0))
            vAF[vAF>(1-dMAF0)]=\
                self.rng.beta(dBetaAF0,dBetaAF0,np.sum(vAF>(1-dMAF0)))
        self.vAF0=vAF
        self.vTau0=(1-vAF)**2
        self.vTau1=1-(vAF**2)
    
    def __draw_betas(self):
        print('Drawing true SNP effects')
        vScaling=(2*self.vAF0*(1-self.vAF0))**(-0.5)
        self.vBetaY=self.rng.normal(size=self.iM)*vScaling
        self.vBetaAM=self.dRhoG*self.vBetaY+\
            ((1-self.dRhoG**2)**0.5)*self.rng.normal(size=self.iM)*vScaling
        self.vBetaGN=self.rng.normal(size=self.iM)*vScaling
    
    def __draw_gen0(self):
        self.__draw_g0()
        self.__draw_y()
    
    def __draw_next_gen(self):
        self.__match()
        self.__mate()
        self.__draw_y()
    
    def __draw_g0(self):
        print('Drawing genotypes founders (=generation 0)')
        self.iT=0
        if self.iSN==0:
            iSN=self.iN
        else:
            iSN=self.iSN
        iB=int(self.iN/iSN)
        iR=(self.iN)%(iSN)
        iBT=iB+(iR>0)
        self.mG=np.empty((1,self.iN,self.iM),dtype=np.int8)
        tCount=tqdm(total=iBT)
        for i in range(iB):
            self.__draw_g0_rows(iSN*i,iSN)
            tCount.update(1)
        if iR>0:
            self.__draw_g0_rows(iSN*iB,iR)
            tCount.update(1)
        tCount.close()
    
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
        print('Drawing genotypes generation '+str(self.iT))
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
        print('Drawing traits generation '+str(self.iT))
        self.mYGN=(self.mG*self.vBetaGN[None,None,:]).mean(axis=2)
        vYGNold=np.zeros(self.iN)
        if self.iT>0:
            vYGNold=(self.dVarGN**0.5)*((self.vYGNold-self.vYGNold.mean())\
                                        /self.vYGNold.std())
        mGY=(self.mG*self.vBetaY[None,None,:]).mean(axis=2)
        mGAM=(self.mG*self.vBetaAM[None,None,:]).mean(axis=2)
        mEY=self.rng.normal(size=(self.iC,self.iN))
        mEAM=self.dRhoE*mEY+\
            ((1-self.dRhoE**2)**0.5)*self.rng.normal(size=(self.iC,self.iN))
        mGY=(mGY-mGY.mean())/mGY.std()
        mGAM=(mGAM-mGAM.mean())/mGAM.std()
        mEY=(mEY-mEY.mean())/mEY.std()
        mEAM=(mEAM-mEAM.mean())/mEAM.std()
        self.mY=(self.dHsqY**0.5)*mGY+((1-self.dHsqY)**0.5)*mEY\
            +vYGNold[None,:]
        self.mYAM=(self.dHsqAM**0.5)*mGAM+((1-self.dHsqAM)**0.5)*mEAM
    
    def __match(self):
        print('Performing assortative mating generation '+str(self.iT))
        iC=1
        if self.iT>0:
            iC=self.iC
        self.vYGNold=np.empty((int(iC*self.iN/2)))
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
            vIndM=vIndM[self.mYAM[i,vIndM].argsort()][vX1rank]
            vIndF=vIndF[self.mYAM[i,vIndF].argsort()][vX2rank]
            self.vYGNold[i*int(self.iN/2):(i+1)*int(self.iN/2)]=\
                self.mYGN[i,vIndM]+self.mYGN[i,vIndF]
            self.vYM[i*int(self.iN/2):(i+1)*int(self.iN/2)]=\
                self.mY[i,vIndM]
            self.vYF[i*int(self.iN/2):(i+1)*int(self.iN/2)]=\
                self.mY[i,vIndF]
            self.mGM[i*int(self.iN/2):(i+1)*int(self.iN/2)]=self.mG[i,vIndM]
            self.mGF[i*int(self.iN/2):(i+1)*int(self.iN/2)]=self.mG[i,vIndF]
    
    def __assign_ids(self):
        if self.iT<1:
            raise SyntaxError('Cannot assign IDs for generation 0')
        self.lSNPs=['SNP_'+str(i+1) for i in range(self.iM)]
        iF=self.mGM.shape[0]
        self.lFID=['Generation'+str(self.iT)+'_Family'+str(i+1)\
              for i in range(iF)]
        self.lIM=[gnames.sMat]*iF
        self.lIF=[gnames.sPat]*iF
        self.lIC=['']*self.iC
        for i in range(self.iC):
            self.lIC[i]=['Child'+str(i+1)]*iF
    
    def __create_dataframes(self):
        if self.iT<1:
            raise SyntaxError('Cannot create DataFrames for generation 0')
        self.__assign_ids()
        miM=pd.MultiIndex.from_arrays([self.lFID,self.lIM],names=gnames.tIDs)
        miF=pd.MultiIndex.from_arrays([self.lFID,self.lIF],names=gnames.tIDs)
        dfG=pd.DataFrame(self.mGM,miM,self.lSNPs)
        dfY=pd.DataFrame(self.vYM,miM,gnames.lPheno)
        dfG=dfG.append(pd.DataFrame(self.mGF,miF,self.lSNPs))
        dfY=dfY.append(pd.DataFrame(self.vYF,miF,gnames.lPheno))
        for i in range(self.iC):
            miC=pd.MultiIndex.from_arrays([self.lFID,self.lIC[i]],\
                                          names=gnames.tIDs)
            dfG=dfG.append(pd.DataFrame(self.mG[i],miC,self.lSNPs))
            dfY=dfY.append(pd.DataFrame(self.mY[i],miC,gnames.lPheno))
        self.dfG=dfG
        self.dfY=dfY
    
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
                dY=self.dfY.loc[self.dfG.index[j]].values[0]
                sIND=sFID+'\t'+sIID+'\t'+sPID+'\t'+sMID+'\t0\t'+str(dY)+'\n'
                oFile.write(sIND)
    
    def __write_bim(self,sName):
        lSNPs=self.dfG.columns.to_list()
        with open(sName+gnames.sBimExt,'w') as oFile:
            for j in range(self.iM):
                sSNP='0\t'+lSNPs[j]+'0\t'+str(j+1)+'\tA\tC\n'
                oFile.write(sSNP)
    
    def __write_bed(self,sName):
        iN=self.dfG.shape[0]
        iB=int(iN/gnames.iNperByte)
        iR=iN%gnames.iNperByte
        iBT=iB+(iR>0)
        mG=np.empty((iBT*gnames.iNperByte,self.iM),dtype=np.uint8)
        mG[0:((iB*gnames.iNperByte)+iR)]=2*(2-self.dfG.values)
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
        Export genotypes to PLINK binary file format
        
        Attributes
        ----------
        sName : string, optional
            prefix for PLINK binary files to generate; default='genotypes'
        """
        if self.iT<1:
            raise SyntaxError('Cannot export to PLINK binary format for '+\
                              'generation 0')
        self.__create_dataframes()
        self.__write_fam(sName)
        self.__write_bim(sName)
        self.__write_bed(sName)
    
    def PerformGWAS(self):
        """
        Perform classical GWAS and within-family GWAS based on offspring data
        """
        if self.iT<1:
            raise SyntaxError('Cannot perform GWAS for generation 0')
        vY=self.mY-self.mY.mean()
        vXTY=(self.mG*vY[:,:,None]).sum(axis=(0,1))
        vXTX=(self.mG**2).sum(axis=(0,1))-\
            self.mG.shape[0]*self.mG.shape[1]*((self.mG.mean(axis=(0,1)))**2)
        self.vBetaGWAS=vXTY/vXTX
        vY=self.mY-self.mY.mean(axis=0)[None,:]
        vXTY=(self.mG*vY[:,:,None]).sum(axis=(0,1))
        vXTX=(((self.mG**2).sum(axis=0))-\
            self.mG.shape[0]*((self.mG.mean(axis=0))**2)).sum(axis=0)
        self.vBetaWF=vXTY/vXTX
    
    def ComputeDiagsGRM(self,dMAF=0.01):
        """
        Compute diagonals of the GRM for the current generation
        
        Attributes
        ----------
        dMAF : float in (0,0.45), optional
            SNPs with an empirical minor-allele frequency below this threshold
            are excluded from calculation of the GRM
        """
        if not(isinstance(dMAF,(int,float))):
            raise ValueError('Minor-allele-frequency threshold not a number')
        if dMAF<=0:
            raise ValueError('Minor-allele-frequency threshold is'+\
                             ' non-positive')
        if dMAF>=gnames.dTooHighMAF:
            raise ValueError('Minor-allele-frequency threshold is'+\
                             ' unreasonably high')
        mG=np.vstack(self.mG)
        vEAF=mG.mean(axis=0)/2
        iMdrop=0
        vDiagDrop=0
        if dMAF>0:
            vDrop=(((vEAF<dMAF)+(vEAF>=(1-dMAF)))>=1)
            iMdrop=vDrop.sum()
            vEAF[vDrop]=0.5
            vDiagDrop=(((mG[:,vDrop]-1)/(0.5**0.5))**2)\
                .sum(axis=1).ravel()
        vDiagAll=(((mG-2*vEAF[None,:])/\
                   (((2*vEAF*(1-vEAF))**0.5)[None,:]))**2).sum(axis=1).ravel()
        iMkeep=self.iM-iMdrop
        vDiag=(vDiagAll-vDiagDrop)/iMkeep
        return vDiag
    
    def Test():
        """
        Function to test if gnames works properly
        """
        iN=1000
        iM=10000
        iT=2
        dHsqAM=1
        dTimeStart=time.time()
        print('Test of gnames with '+str(iN)+' founders and '+str(iM)+' SNPs')
        print('For '+str(iT)+' offspring generations')
        print('With heritability of assortative-mating trait 100%')
        print('With 2 children per mating pair')
        print('Initialising simulator')
        simulator=gnames(iN,iM,dHsqAM=dHsqAM)
        print('Highest diagonal element of GRM for founders = '+\
              str(round(max(simulator.ComputeDiagsGRM()),3)))
        print('Simulating data for '+str(iT)+' subsequent generations')
        simulator.Simulate(iT)
        print('Highest diagonal element of GRM after '+str(iT)+\
              ' generations = '+str(round(max(simulator.ComputeDiagsGRM()),3)))
        dTime=time.time()-dTimeStart
        print('Writing PLINK binary files (genotypes.bed, .bim, .fam)')
        simulator.MakeBed()
        print('Runtime: '+str(round(dTime,3))+' seconds')
