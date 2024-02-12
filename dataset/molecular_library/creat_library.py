import os
import pandas as pd

PATH = './similar/2D_85/'
file_list = []
for _,_,file_name in os.walk(PATH):
    file_list.append(file_name)
file_list = file_list[0]

#83 files in total
library = pd.read_csv(PATH+file_list[0],header=0,usecols=['cid','cmpdname','isosmiles'])
for file_name in file_list[1:]:
    temp = pd.read_csv(PATH+file_name,header=0,usecols=['cid','cmpdname','isosmiles'])
    library = pd.concat([library,temp])
    library = library.drop_duplicates(subset=['cid'])

with pd.ExcelWriter('./molecular_library_2D85.xlsx') as writer:
    library.to_excel(writer)