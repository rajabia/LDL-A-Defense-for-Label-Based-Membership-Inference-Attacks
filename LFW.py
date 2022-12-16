
from os import listdir
import glob

import shutil 

data_path='data/lfw_original/'

file_list = listdir(data_path)
count=0
for f in file_list:
	if  len(glob.glob(data_path+f+'/*.jpg'))>=40:

		shutil.copytree(data_path+f, 'data/lfw/'+f, copy_function = shutil.copy)
		count+=1
	print(f, len(glob.glob(data_path+f+'/*.jpg')))

print(count)