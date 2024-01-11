import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd

url = 'https://www.turkishairlines.com/tr-int/bilgi-edin/check-in-sorular/index.html'
driver = webdriver.Chrome()
driver.get(url)
content = driver.page_source


