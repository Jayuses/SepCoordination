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

#url = 'https://www.chemicalbook.com/ProductCatalog_EN/1329.htm' #Products of Organic phosphine compound
#url = 'https://www.chemicalbook.com/ProductCatalog_EN/1313.htm' #Nitrogen Compounds
#url = 'https://www.chemicalbook.com/ProductCatalog_EN/1317.htm' #Ethers and derivatives
#url = 'https://www.chemicalbook.com/ProductCatalog_EN/1318.htm'  #Aldehydes
#url = 'https://www.chemicalbook.com/ProductCatalog_EN/1324.htm'  #Heterocyclic compounds
url = 'https://www.chemicalbook.com/ProductCatalog_EN/1328.htm'  #Organosulfur
# urls = [url[:51]+'-'+str(i)+'.htm' for i in range(2,20)]
urls = []
urls.insert(0,url)

driver = webdriver.Edge(EdgeChromiumDriverManager().install())
cas = []
name = []
mf = []
mw = []
Melting_point = []
Boiling_point = []
density = []
count = 0

for ur in urls:
    driver.get(ur)
    table = driver.find_elements_by_xpath("//td")
    driver.implicitly_wait(5)

    for ind,tab in enumerate(table):
        if ind % 4 == 1:
            name.append(tab.text)
        elif ind % 4 == 2:
            cas.append(tab.text)
        elif ind % 4 == 3:
            mf.append(tab.text)
    
df = pd.DataFrame({'Name':name,'CAS':cas,'MF':mf})

for ur in urls:
    driver.get(ur)
    table = driver.find_elements_by_xpath("//td")
    driver.implicitly_wait(5)
    num = len(table)

    for i in range(num):
        if i % 4 == 1:
            element = table[i].find_element_by_tag_name("a")
            mname = element.text
            element.click()
            driver.implicitly_wait(5)
            temp = driver.find_element_by_xpath("//td[text()='MW:']/following-sibling::td")
            mw.append(temp.text)
            temp = driver.find_elements_by_xpath("//*[@id='ContentPlaceHolder1_ProductChemPropertyA']//td")
            flag = [0,0,0]
            for ind,tem in enumerate(temp):
                if tem.text == 'Melting point ':
                    Melting_point.append(temp[ind+1].text)
                    flag[0] = 1
                elif tem.text == 'Boiling point ':
                    Boiling_point.append(temp[ind+1].text)
                    flag[1] = 1
                elif tem.text == 'density ':
                    density.append(temp[ind+1].text)
                    flag[2] = 1
            if flag[0] == 0:
                Melting_point.append('')
            if flag[1] == 0:
                Boiling_point.append('')
            if flag[2] == 0:
                density.append('')
            
            count += 1
            print(f'{count}:{mname}')
            driver.back()
            table = driver.find_elements_by_xpath("//td")
            driver.implicitly_wait(5)
        else:
            continue

df['mw'],df['Melting_point'],df['Boiling_point'],df['density'] = mw,Melting_point,Boiling_point,density

url='https://pubchem.ncbi.nlm.nih.gov/search/#query=cas'
cases = cas
driver = webdriver.Edge(EdgeChromiumDriverManager().install())
ua=UserAgent()
smiles = []

for c in cases:
    if not isinstance(c,str):
        continue
    cas_url = re.sub(r'cas',c.strip(),url)
    headers = {"User-Agent": ua.random}
    driver.get(cas_url)
    driver.implicitly_wait(5)
    try:
        temp = driver.find_element_by_xpath(
            "//*[text()='Best Match']/following-sibling::div//*[text()='Isomeric SMILES']/following-sibling::span/span")
        smiles.append(temp.text)
    except:
        smiles.append('')
    print(c)
    

df['smiles'] = smiles
with pd.ExcelWriter('./chembook.xlsx',mode='a') as writer:
    df.to_excel(writer,'Organosulfur',index=False)