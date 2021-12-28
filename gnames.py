import time
import pandas as pd
import numpy as np

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
    '''
    dTooHighMAF=0.45
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
        self.iT=0
        self.__draw_g0()
        self.__draw_y()
        self.__match()
    
    def __draw_next_gen(self):
        self.iT+=1
        self.__mate()
        self.__draw_y()
        self.__match()
    
    def __draw_g0(self):
        print('Drawing genotypes founders (=generation 0)')
        if self.iSN==0:
            iSN=self.iN
        else:
            iSN=self.iSN
        iB=int(self.iN/iSN)
        iR=(self.iN)%(iSN)
        if iR>0:
            iT=iB+1
        else:
            iT=iB
        self.mG=np.empty((1,self.iN,self.iM),dtype=np.int8)
        for i in range(iB):
            print('-> block '+str(i+1)+' out of '+str(iT))
            self.__draw_g0_rows(iSN*i,iSN)
        if iR>0:
            print('Block '+str(iT)+' out of '+str(iT))
            self.__draw_g0_rows(iSN*iB,iR)
    
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
        print('Drawing genotypes children for generation '+str(self.iT))
        self.iN=self.mGM.shape[0]
        self.mG=np.empty((self.iC,self.iN,self.iM),dtype=np.int8)
        mCM=np.zeros((self.iN,self.iM),dtype=np.int8)
        mCF=np.zeros((self.iN,self.iM),dtype=np.int8)
        mCM[self.mGM==2]=1
        mCF[self.mGF==2]=1
        for i in range(self.iC):
            print('-> for set of children '+str(i+1)+' out of '+str(self.iC))
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
        mGY=(mGY-(mGY.mean(axis=1)[:,None]))/(mGY.std(axis=1)[:,None])
        mGAM=(mGAM-(mGAM.mean(axis=1)[:,None]))/(mGAM.std(axis=1)[:,None])
        mEY=(mEY-(mEY.mean(axis=1)[:,None]))/(mEY.std(axis=1)[:,None])
        mEAM=(mEAM-(mEAM.mean(axis=1)[:,None]))/(mEAM.std(axis=1)[:,None])
        self.mY=(self.dHsqY**0.5)*mGY+((1-self.dHsqY)**0.5)*mEY\
            +vYGNold[None,:]
        self.mYAM=(self.dHsqAM**0.5)*mGAM+((1-self.dHsqAM)**0.5)*mEAM
    
    def __match(self):
        print('Performing assortative mating generation '+str(self.iT))
        iC=1
        if self.iT>0:
            iC=self.iC
        self.vYGNold=np.empty((int(iC*self.iN/2)))
        self.mGM=np.empty((int(iC*self.iN/2),self.iM))
        self.mGF=np.empty((int(iC*self.iN/2),self.iM))
        for i in range(iC):
            print('-> for group '+str(i+1)+' out of '+str(iC))
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
            self.mGM[i*int(self.iN/2):(i+1)*int(self.iN/2)]=self.mG[i,vIndM]
            self.mGF[i*int(self.iN/2):(i+1)*int(self.iN/2)]=self.mG[i,vIndF]
    
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
        vEAF=np.vstack(self.mG).mean(axis=0)/2
        iMdrop=0
        vDiagDrop=0
        if dMAF>0:
            vDrop=(((vEAF<dMAF)+(vEAF>=(1-dMAF)))>=1)
            iMdrop=vDrop.sum()
            vEAF[vDrop]=0.5
            vDiagDrop=(((np.vstack(self.mG)[:,vDrop]-1)/(0.5**0.5))**2)\
                .sum(axis=1).ravel()
        vDiagAll=(((np.vstack(self.mG)-2*vEAF[None,None,:])\
                   /(((2*vEAF*(1-vEAF))**0.5)[None,None,:]))**2)\
            .sum(axis=2).ravel()
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
        dTimeStart=time.time()
        print('Test of gnames with '+str(iN)+' founders and '+str(iM)+' SNPs')
        print('For '+str(iT)+' offspring generations')
        print('With 2 children per mating pair')
        print('Initialising simulator')
        simulator=gnames(iN,iM)
        print('Highest diagonal element of GRM for founders = '+\
              str(round(max(simulator.ComputeDiagsGRM()),3)))
        print('Simulating data for '+str(iT)+' subsequent generations')
        simulator.Simulate(iT)
        print('Highest diagonal element of GRM after '+str(iT)+\
              ' generations = '+str(round(max(simulator.ComputeDiagsGRM()),3)))
        dTime=time.time()-dTimeStart
        print("Runtime: "+str(round(dTime,3))+" seconds")
