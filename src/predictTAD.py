from sys import argv
from utils import predictTADmultiRes_,predictTADmultiRes 
import time      
                 
script,model,latentTADFiles,resolutions,n,outPrefix = argv
modelLists = model.split(',')
latentTADLists = latentTADFiles.split(',')
reslist = [int(x) for x in resolutions.split(',')]
n = int(n)

if n == 0:
    predictTADmultiRes_(modelLists,latentTADLists,reslist,outPrefix)
else:
    predictTADmultiRes(modelLists,latentTADLists,reslist,n,outPrefix)
