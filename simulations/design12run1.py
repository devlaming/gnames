import numpy as np
from gnames import gnames
#DESIGN 12: Various levels of assortative mating but no genetic nurture,
#and without per-generation rescaling of heritable component,
#allowing panmictic, contemporary, and equilibrium heritability to differ
iDESIGN=12
bRescaling=False
iNGWAS=16000
iNPRED=16000
iM=5000
dHsqY=0.25
dPropGN=0
vRhoAM=np.array((0,0.25,0.5,0.75,1))
dCorrYAM=1
iC=2
iT=10
dGWASMAF=0.01
iSETTINGS=len(vRhoAM)
#RUN NUMBER
iR=1
for iSetting in range(iSETTINGS):
    #get unique seed for given combination of run, setting, and design
    #bits 1-4: for design numbers between 0 and 15 (=2**(4-0)-1)
    #bits 5-8: for setting numbers between 0 and 15 (=2**(8-4)-1)
    #bits 9-.: for run numbers between 0 and 16777215 (=2**(32-8)-1)
    iThisSeed=(2**0)*iDESIGN+(2**4)*iSetting+(2**8)*iR
    dRhoAM=vRhoAM[iSetting]
    iN=(2*iC*iNGWAS)+iNPRED
    sName='DESIGN.'+str(iDESIGN)+'.RHO_AM.'+'{:.2f}'.format(dRhoAM)+'.RUN.'+str(iR)
    simulator=gnames(iN,iM,iC,dHsqY,dPropGN,dCorrYAM,dRhoAM,iSeed=iThisSeed,bRescale=bRescaling)
    simulator.Simulate(iT)
    simulator.MakeThreePGIs(sName,iNGWAS,iNPRED,dGWASMAF)
