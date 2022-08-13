#write the chunks of pages

# !pip3 install -U selenium
# !pip3 install webdriver-manager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import csv
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.set_window_size(800,600)
def get(url):
    f=open('amazon_reviews.csv','w')
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['name','fresh'])

    driver.get(url)

    while True:
        # driver.execute_script('window,scrollTo(0,document.body.scrollHeight)')
        l=driver.find_elements(by=By.CSS_SELECTOR,value='[class="row review_table_row"]')
        for v in l:
            r=[]
            r.append(v.find_element(by=By.CSS_SELECTOR,value='[href*="/critics/"]').text)

            temp=v.find_element(by=By.CSS_SELECTOR,value='[class*="review_icon icon small "]').get_attribute('class')
            r.append(temp[temp.rindex(' ')+1:])
            writer.writerow(r)

        b=WebDriverWait(driver,5).until(EC.presence_of_element_located((By.CLASS_NAME,"prev-next-paging__button-right")))
        if 'hide' in b.get_attribute('class'):break
        b.click()
        time.sleep(1)

get('https://www.rottentomatoes.com/m/exodus_gods_and_kings/reviews')
driver.close()

