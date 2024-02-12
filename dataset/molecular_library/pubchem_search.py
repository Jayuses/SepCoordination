import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from fake_useragent import UserAgent
import pandas as pd
import numpy as np
import time

url='https://pubchem.ncbi.nlm.nih.gov/search/#query=smile&tab=similarity&similaritythreshold=85'
search_path = '../Data/ourdata/Cu_graph.xlsx'
search_smiles = pd.read_excel(search_path,0,usecols='AQ').to_numpy()
search_smiles = np.squeeze(search_smiles).tolist()

driver = webdriver.Edge(EdgeChromiumDriverManager().install())

TANIMOTO = 85
ua=UserAgent()

for smi in search_smiles:
    smi_url = re.sub(r'smile',smi.strip(),url)
    headers = {"User-Agent": ua.random}
    driver.get(smi_url)
    driver.implicitly_wait(5)
    # result = driver.find_elements(By.XPATH,"//*[text()='0 results found...']")
    # if len(result)==0:
    WebDriverWait(driver,timeout=20).until(lambda d:d.find_element_by_id('Download'))
    download = driver.find_element_by_id('Download')
    download.click()
    WebDriverWait(driver,timeout=20).until(lambda d:d.find_element(By.LINK_TEXT,'CSV'))
    csv = driver.find_element(By.LINK_TEXT,'CSV')
    csv.click()
    print(f'{smi} done')
    time.sleep(5)
    # else:
    #     print(f"{smi} can't found")