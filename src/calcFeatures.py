from sys import argv
from utils import callTads

script,cool,resolutions,outPrefix = argv
resList = [int(x) for x in resolutions.split(',')]

calcDI_check=True
for res in resList:
    callTads(cool,res,outPrefix,calcDI_check)
    calcDI_check=False
