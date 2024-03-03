from sys import argv
from utils import predictTAD,predictTADmultiRes

script,model,latentTADFiles,resolutions,outPrefix = argv
modelLists = model.split(',')
latentTADLists = latentTADFiles.split(',')
reslist = [int(x) for x in resolutions.split(',')]

predictTADmultiRes(modelLists,latentTADLists,reslist,outPrefix)
