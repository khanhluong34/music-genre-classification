import os
from utils import *
from selenium import webdriver

from selenium import webdriver

from selenium.webdriver.chrome.options import Options


from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

# Add arguments to chrome_options

chrome_options = Options()
chrome_options.add_argument("--incognito")   #switch to incognito tab
chrome_options.add_argument("--window-size=1920x1080")   #set window size
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")   #disable gpu
chrome_options.add_argument("user-agent=Chrome/80.0.3987.132")
chrome_options.add_argument("--headless")   #now showing the tab while running

# Get song links 

# The path to folder containing txt files of genre urls
genre_urls_path = './genre_urls'
genre_urls = get_genre_urls(genre_urls_path)



