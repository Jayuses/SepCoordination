from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np

PATH = 'http://www.chemspider.com/'
search_path = './exceed.xlsx'
search_data = pd.read_excel(search_path,0,usecols='A,B,C')
search_smiles = search_data['isosmiles'].to_numpy().tolist()

driver = webdriver.Edge(EdgeChromiumDriverManager().install())
driver.get(PATH)
driver.implicitly_wait(10)

cas_list = []
density = []
melting_point = []
boiling_point = []
prop_dict = []

for smi in search_smiles:
    prop = {}
    box = driver.find_element(By.XPATH,"//*[@id='pnlSiteSearch']//input")
    button = driver.find_element(By.XPATH,"//*[@id='pnlSiteSearch']//button")
    box.send_keys(smi)
    button.click()
    driver.implicitly_wait(10)

    if 'Found 0 results' in driver.page_source:
        prop_dict.append(prop)
        cas_list.append('')
    else:
        if driver.find_elements(By.XPATH,"//*[@class='search-id-column']/a") == []:
            pass
        else:
            link = driver.find_elements(By.XPATH,"//*[@class='search-id-column']/a")[1]
            driver.execute_script("arguments[0].click();", link)
            driver.implicitly_wait(10)
        
        cas = driver.find_element(By.XPATH,"//div[1][@class='syn']").text
        cas_list.append(cas)
        properties = driver.find_element(By.LINK_TEXT,"Properties")
        driver.execute_script("arguments[0].click();", properties)
        prop_title = driver.find_elements(By.XPATH,"//*[@class='AspNet-FormView-Data']//td[@class='prop_title']")
        prop_value = driver.find_elements(By.XPATH,"//*[@class='AspNet-FormView-Data']//td[@class='prop_value_nowrap' or @class='prop_value']")
        for ind,title in enumerate(prop_title):
            prop[title.text] = prop_value[ind].text
        prop_dict.append(prop)
    
    print(f'{smi} done!')