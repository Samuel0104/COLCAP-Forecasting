from driver_setup import *

# Website Initialization
driver.get("https://www.portafolio.co/buscar?q=")
time.sleep(2)
titles = []
dates = []
texts = []

# Search by SECTIONS
sections = ["Tecnología", "Energía", "Economía", "Indicadores Económicos",
            "Internacional", "Inversión", "Negocios"]
for section in sections:
    button1 = driver.find_element(By.NAME, "category")
    button1.send_keys(section)
    button2 = driver.find_element(By.ID, "adv_search_submit")
    driver.implicitly_wait(2)
    ActionChains(driver).move_to_element(button2).click(button2).perform()
    print(f"{section}:")

    # Search by KEYWORDS
    with open("keywords.txt") as file:
        keywords = file.read().splitlines()
    for keyword in keywords:
        bar = driver.find_element(By.XPATH, '//*[@id="main-container"]/div[2]/form/input')
        bar.send_keys(keyword)
        button3 = driver.find_element(By.CLASS_NAME, "buscar")
        driver.implicitly_wait(10)
        ActionChains(driver).move_to_element(button3).click(button3).perform()
        print(f"\t{keyword}")
        
        # Data collection
        temp1 = [title.text for title in driver.find_elements(By.CLASS_NAME, "listing-title")]
        temp2 = [date.text for date in driver.find_elements(By.CLASS_NAME, "listing-time")]
        links = driver.find_elements(By.CLASS_NAME, "news-title")
        for i, link in enumerate(links):
            url = link.get_attribute("href")
            req = requests.get(url)
            soup = bs4.BeautifulSoup(req.text, features="lxml")
            content = soup.find(class_="article-content")
            if content:
                titles.append(temp1[i])
                dates.append(temp2[i])
                texts.append(content.get_text())

# Dataframe
df = pd.DataFrame({
    "headline": titles,
    "date": dates,
    "content": texts})
df.to_csv("portafolio.csv")
