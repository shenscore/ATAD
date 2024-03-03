from utils import align_boundary
from sys import argv

script,res,di_check,bedpeFiles,out = argv
domainList = bedpeFiles.split(',')
res = int(res)

align_boundary(domainList,di_check,out,distance=6,res=res)
