import os
from os import listdir
from os.path import isdir

sourceDir = "dataset\\txt"

i = 0

for file_name in os.listdir(sourceDir):
    i=str(i)
    source = sourceDir + "\\" + file_name
    dest =  sourceDir + "\\" + i + ".txt"
    os.rename(source, dest)
    i=int(i)
    i=i+1
    