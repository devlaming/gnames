import numpy as np
from gnames import gnames
#DESIGN 0: Various GWAS sample sizes, no assortative mating, no genetic nurture
iDESIGN=0
vNGWAS=np.array((1000,2000,5000,10000))
iNPRED=20000
iSETTINGS=len(vNGWAS)+1
dRhoAM=0
dPropGN=0
#OTHER SPECS
iC=2
iT=10
iM=2000
dHsqY=0.25
dCorrYAM=1
dGWASMAF=0.01
#RUN NUMBER
iR=1
for iSetting in range(1,iSETTINGS):
    #get unique seed for given combination of run, setting, and design
    #bits 1-4: for iDESIGN between 1 and 15 (=2**4-1)
    #bits 5-8: for settings between 1 and 15 (=2**(8-4)-1)
    #bits 9-.: for run between 1 and 16777215 (=2**(32-8)-1)
    iThisSeed=(2**0)*iDESIGN+(2**4)*iSetting+(2**8)*iR
    iNGWAS=int(vNGWAS[iSetting-1])
    iN=(2*iC*iNGWAS)+iNPRED
    sName='DESIGN.'+str(iDESIGN)+'.NGWAS.'+str(iNGWAS)+'.RUN.'+str(iR)
    simulator=gnames(iN,iM,iC,dHsqY,dPropGN,dCorrYAM,dRhoAM,iSeed=iThisSeed)
    simulator.Simulate(iT)
    simulator.MakeThreePGIs(sName,iNGWAS,iNPRED,dGWASMAF)
