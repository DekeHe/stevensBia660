#
# !pip3 install -U selenium
# !pip3 install webdriver-manager


from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import re, time,csv
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def getAmazonReviews(driver, url, outpath):
    # open a new csv writer
    fw = open(outpath, 'w', encoding='utf8')
    writer = csv.writer(fw, lineterminator='\n')
    writer.writerow(['title', 'date', 'content', 'rating'])

    driver.get(url)  # visit the reviews url for the given product

    while True:  # keep going until there are no more review pages

        # scroll down
        driver.execute_script('window,scrollTo(0,document.body.scrollHeight)')

        # get all the reviews in the page
        reviews = driver.find_elements(by=By.CSS_SELECTOR, value='[data-hook="review"]')

        for review in reviews:  # for each review

            # initialize key attributes
            rating, content, title, date, = 'NA', 'NA', 'NA', 'NA'

            try:  # try to find the title box
                titleBox = review.find_element(by=By.CSS_SELECTOR, value='[data-hook="review-title"]')
            except:
                titleBox = None

                # box found, extract text
            if titleBox: title = titleBox.text

            try:  # try to find the date box
                dateBox = review.find_element(by=By.CSS_SELECTOR, value='[data-hook="review-date"]')
            except:
                dateBox = None

            # box found, extract text
            if dateBox: date = dateBox.text

            try:  # try to find the rating box
                ratingBox = review.find_element(by=By.CSS_SELECTOR, value='[data-hook*="review-star-rating"]')
            except NoSuchElementException:
                ratingBox = None

            # box found
            if ratingBox:
                ratingInfo = ratingBox.get_attribute('class')  # get the text of class attribute

                rating = re.search('a-star-(\d)', ratingInfo)  # look for the star rating from the class text

                rating = rating.group(1)  # extract the star rating

            try:  # try to find the content box
                contentBox = review.find_element(by=By.CSS_SELECTOR, value='[data-hook="review-body"]')
            except NoSuchElementException:
                contentBox = None

            # box found, extract text
            if contentBox: content = contentBox.text

            # write a new row
            writer.writerow([title, date, content, rating])

        # wait until the next Button loads
        nextButton = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'a-last')))

        if 'a-disabled' in nextButton.get_attribute(
                'class'):  # final page reached, 'next' button is disabled on this page
            break

        # click on the next Button
        nextButton.click()

        # wait for a few seconds
        time.sleep(3)

    fw.close()

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.maximize_window()

url='https://www.amazon.com/Wireless-Headphones-Watching-Transmitter-Rechargeable/product-reviews/B099DD32XN/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'

outpath='amazon_reviews.csv'


getAmazonReviews(driver,url,outpath)