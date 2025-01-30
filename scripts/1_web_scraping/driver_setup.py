# General libraries
import time
import pandas as pd
import re

# Web scraping
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

# System
import os
import urllib3
os.environ["CURL_CA_BUNDLE"] = ""
urllib3.disable_warnings()

# Driver setup
opts = webdriver.ChromeOptions()
opts.add_argument("--headless")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
prefs = {"profile.managed_default_content_settings.images": 2}
opts.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome(options=opts)
