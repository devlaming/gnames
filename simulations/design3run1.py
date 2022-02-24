import numpy as np
from gnames import gnames
#DESIGN 3: Genetic nurture + Various hold-out sample sizes
iDESIGN=3
iNGWAS=16000
vNPRED=np.array((1000,2000,4000,8000,16000))
iM=5000
dHsqY=0.2
dPropGN=0.1
dRhoAM=0
dCorrYAM=1
iC=2
iT=10
dGWASMAF=0.01
iSETTINGS=len(vNPRED)
#RUN NUMBER
iR=1
for iSetting in range(iSETTINGS):
    #get unique seed for given combination of run, setting, and design
    #bits 1-4: for design numbers between 0 and 15 (=2**(4-0)-1)
    #bits 5-8: for setting numbers between 0 and 15 (=2**(8-4)-1)
    #bits 9-.: for run numbers between 0 and 16777215 (=2**(32-8)-1)
    iThisSeed=(2**0)*iDESIGN+(2**4)*iSetting+(2**8)*iR
    iNPRED=int(vNPRED[iSetting])
    iN=(2*iC*iNGWAS)+iNPRED
    sName='DESIGN.'+str(iDESIGN)+'.NHOLDOUT.'+str(iNPRED)+'.RUN.'+str(iR)
    simulator=gnames(iN,iM,iC,dHsqY,dPropGN,dCorrYAM,dRhoAM,iSeed=iThisSeed)
    simulator.Simulate(iT)
    simulator.MakeThreePGIs(sName,iNGWAS,iNPRED,dGWASMAF)
