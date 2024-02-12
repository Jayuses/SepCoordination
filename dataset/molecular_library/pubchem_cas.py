import regex as re
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from fake_useragent import UserAgent
import pandas as pd
import numpy as np
import time

url='https://pubchem.ncbi.nlm.nih.gov/search/#query='

search_path = './exceed.xlsx'
search_data = pd.read_excel(search_path,sheet_name=2,usecols='A,B,C',header=0)
search_smiles = search_data['isosmiles'].to_numpy().tolist()

driver = webdriver.Edge(EdgeChromiumDriverManager().install())

ua=UserAgent()

def find_cas(smi_list):
    cas_list = []
    for smi in smi_list:
        smi_url = url + smi
        headers = {"User-Agent": ua.random}
        driver.get(smi_url)
        driver.implicitly_wait(5)
        comp = driver.find_element_by_link_text('Summary')
        comp.click()
        time.sleep(5)

        identities = driver.find_elements_by_xpath("//ul[@class='break-words space-y-1 list-none leading-tight']/li")
        if len(identities) == 0:
            cas_list.append('null')
            print(f'{smi}:{cas_list[-1]}')
            time.sleep(10)
            continue
        for ind,iden in enumerate(identities):
            sep_iden = iden.text.split(sep='-')
            if len(sep_iden) == 3 and (sep_iden[0].isdigit() and sep_iden[1].isdigit()):
                cas_list.append(iden.text)
                break
            else:
                if ind == len(identities) - 1:
                    cas_list.append('null')
                continue
        print(f'{smi}:{cas_list[-1]}')
        time.sleep(10)

    return pd.DataFrame({'cas':cas_list})

if __name__ == '__main__':
    cas_list1 = find_cas(search_smiles[50:100])
    # search_data = pd.read_excel(search_path,sheet_name=1,usecols='A,B,C',header=0)
    # search_smiles = search_data['isosmiles'].to_numpy().tolist()
    # cas_list2 = find_cas(search_smiles)
    with pd.ExcelWriter('./temp.xlsx', mode='a') as writer:
        cas_list1.to_excel(writer, sheet_name='2D85', index=False)
        # cas_list2.to_excel(writer, sheet_name='3D85', index=False)