from driver_setup import *

titles = []
dates = []
links = []

with open("keywords.txt") as file:
    keywords = file.read().splitlines()

for keyword in keywords:
    driver.get(f"https://www.larepublica.co/buscar?term={keyword}&formatsRaw=Art%C3%ADculo&feedTypes=articulos&from=2015-01-01T00%3A00%3A00-05%3A00&to=2024-12-31T00%3A00%3A00-05%3A00")
    time.sleep(2)
    print(f"{keyword}")
    
    try: button = driver.find_element(By.XPATH, '//*[@id="vue-container"]/div[2]/div[2]/div[2]/div/div/button/a')
    except: button = None
    while button:
        driver.implicitly_wait(10)
        ActionChains(driver).move_to_element(button).click(button).perform()
        try: button = driver.find_element(By.XPATH, '//*[@id="vue-container"]/div[2]/div[2]/div[2]/div/div/button/a')
        except: button = None
    time.sleep(10)
    
    a = driver.find_elements(By.TAG_NAME, "h3")
    for title in a:
        titles.append(title.text)
    
    b = driver.find_elements(By.CLASS_NAME, "date")
    for date in b:
        dates.append(date.text)
    
    c = driver.find_elements(By.CLASS_NAME, "result")
    for link in c:
        url = link.get_attribute("href")
        links.append(url)
driver.quit()

# Dataframe
df = pd.DataFrame({
    "headline": titles,
    "date": dates,
    "link": links})
df.drop_duplicates("link", inplace=True, ignore_index=True)
df.to_csv("../../data/larepublica.csv", index=False)
