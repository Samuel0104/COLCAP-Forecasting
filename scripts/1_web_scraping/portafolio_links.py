from driver_setup import *

# Website Initialization
driver.get("https://www.portafolio.co/buscar?q=&sort_field=publishedAt&category=&publishedAt%5Bfrom%5D=2015-01-01&publishedAt%5Buntil%5D=2024-12-31&contentTypes%5B%5D=article&contentTypes%5B%5D=gallery&contentTypes%5B%5D=video")
time.sleep(2)
titles = []
dates = []
links = []
texts = []

sections = ["Energía", "Economía", "Indicadores Económicos",
            "Internacional", "Inversión", "Negocios"]
with open("keywords.txt") as file:
    keywords = file.read().splitlines()

# Search by SECTIONS
for section in sections:
    # Form for choosing the section
    button1 = driver.find_element(By.NAME, "category")
    button1.send_keys(section)
    # "Submit" button
    button2 = driver.find_element(By.ID, "adv_search_submit")
    driver.implicitly_wait(2)
    ActionChains(driver).move_to_element(button2).click(button2).perform()
    print(f"{section}:")

    # Search by KEYWORDS
    for keyword in keywords:
        # Space to write the topic to search for
        bar = driver.find_element(By.XPATH, '//*[@id="main-container"]/div[2]/form/input')
        bar.send_keys(keyword)
        # "Search" button
        button3 = driver.find_element(By.CLASS_NAME, "buscar")
        driver.implicitly_wait(10)
        ActionChains(driver).move_to_element(button3).click(button3).perform()
        print(f"\t{keyword}")

        # Data collection
        try: pages = driver.find_element(By.CLASS_NAME, "pagination")
        except: pages = None
        count = 0
        col1_prev = []
        while count < 1000:
            count += 1
            print(f"\t\t{count}")
            col1 = [title.text for title in driver.find_elements(By.CLASS_NAME, "listing-title")]
            if col1 == col1_prev:
                break
            titles.extend(col1)
            for date in driver.find_elements(By.CLASS_NAME, "listing-time"):
                temp = date.text.split()
                dates.append(" ".join(temp[3:7]))
            for link in driver.find_elements(By.CLASS_NAME, "news-title"):
                url = link.get_attribute("href")
                links.append(url)
            if not pages:
                break
            col1_prev = col1
            button4 = driver.find_element(By.CLASS_NAME, "next")
            button4.click()

# Dataframe
df = pd.DataFrame({
    "headline": titles,
    "date": dates,
    "link": links})
df.dropna(inplace=True)
df.drop_duplicates("link", inplace=True, ignore_index=True)
df.to_csv("portafolio.csv")
