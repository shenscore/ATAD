from sys import argv
from utils import predictTAD,predictTADmultiRes

script,model,latentTADFiles,resolutions,n,outPrefix = argv
modelLists = model.split(',')
latentTADLists = latentTADFiles.split(',')
reslist = [int(x) for x in resolutions.split(',')]
n = int(n)

predictTADmultiRes(modelLists,latentTADLists,reslist,n,outPrefix)
