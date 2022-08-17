import numpy as np
from gnames import gnames
#DESIGN 11: Heritability of 50%, considering various discovery and hold-out sample sizes
iDESIGN=11
vNGWAS=(1000,7500)
vNPRED=(1000,4000,16000)
iM=5000
dHsqY=0.50
dPropGN=0
dRhoAM=0
dCorrYAM=1
iC=2
iT=10
dGWASMAF=0.01
iSetting=0
#RUN NUMBER
iR=1
for iNGWAS in vNGWAS:
    for iNPRED in vNPRED:
        #get unique seed for given combination of run, setting, and design
        #bits 1-4: for design numbers between 0 and 15 (=2**(4-0)-1)
        #bits 5-8: for setting numbers between 0 and 15 (=2**(8-4)-1)
        #bits 9-.: for run numbers between 0 and 16777215 (=2**(32-8)-1)
        iThisSeed=(2**0)*iDESIGN+(2**4)*iSetting+(2**8)*iR
        iN=(2*iC*iNGWAS)+iNPRED
        sName='DESIGN.'+str(iDESIGN)+'.HSQ.'+str(dHsqY)+'.NGWASPOOLED.'+str(2*iNGWAS)+'.NHOLDOUT.'+str(iNPRED)+'.RUN.'+str(iR)
        simulator=gnames(iN,iM,iC,dHsqY,dPropGN,dCorrYAM,dRhoAM,iSeed=iThisSeed)
        simulator.Simulate(iT)
        simulator.MakeThreePGIs(sName,iNGWAS,iNPRED,dGWASMAF)
        iSetting+=1
