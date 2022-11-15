import os
from os import listdir
from os.path import isdir

sourceDir = "training"
for subdir in listdir(sourceDir):
    path = sourceDir + "\\" + subdir + "\\"
    if isdir(path):
        for file_name in os.listdir(path):
    # Construct old file name
            source = path + file_name

            # Adding the count to the new file name and extension
            destination = subdir + file_name

            os.rename(source, destination)
    else:
        continue
    