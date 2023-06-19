import os
from utils import get_genre_urls
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

chrome_options = Options()
# chrome_options.add_argument("--incognito")   #switch to incognito tab
# chrome_options.add_argument("--window-size=1920x1080")   #set window size
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-gpu")   #disable gpu
# chrome_options.add_argument("user-agent=Chrome/80.0.3987.132")
chrome_options.add_argument("--headless=new")   #now showing the tab while running

executable_path = './chrome.exe'
chrome_service = Service(executable_path)

driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

def get_song_urls(genre_urls: dict):  

    PATH = './song_urls'
    if not os.path.exists(PATH):
        os.makedirs(PATH)


    for genre, urls in genre_urls.items():
        category = str(genre) + '.txt'
        category_path = os.path.join(PATH, category)      
        with open(category_path, 'w', encoding='utf-8') as f:
            for url in urls:
                driver.get(url)
                sleep(2)     
                bottom_sentinel = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.XPATH, "//div[@data-testid='bottom-sentinel']")))
                song_arr = []
                reached_page_end = False

                while not reached_page_end:
                    sleep(1)
                    table = BeautifulSoup(driver.page_source, 'html.parser')
                    song_list = table.findAll("div", {'data-testid': 'tracklist-row'})
                    test_end_page = 0

                    for song in song_list: 
                        try:
                            song_info = song.find('a', {'class': 't_yrXoUO3qGsJS4Y6iXX'})
                            song_link = 'https://open.spotify.com' + str(song_info.get('href'))
                            song_name = song_info.find('div', {'class': 'Type__TypeElement-sc-goli3j-0 kHHFyx t_yrXoUO3qGsJS4Y6iXX standalone-ellipsis-one-line'})
                            sn = ''
                            if len(song_name.contents) != 0 :
                                sn = song_name.contents[0]

                            if song_link not in song_arr:
                                test_end_page += 1
                                song_arr.append(str(song_link))
                                f.write(str(song_link + ' ' + sn + '\n'))

                        except Exception as e:
                            print(e)
                            pass

                    bottom_sentinel.location_once_scrolled_into_view
                    driver.implicitly_wait(15)

                    if test_end_page == 0:
                        reached_page_end = True
                    # else:
                    #     print('pass')
        
    driver.quit()


if __name__ == '__main__':
    genre_urls_path = './genre_urls'
    genre_urls = get_genre_urls(genre_urls_path)
    get_song_urls(genre_urls)