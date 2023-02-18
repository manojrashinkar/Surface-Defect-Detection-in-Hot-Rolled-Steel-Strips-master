import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://byjus.com/maths/trigonometry-formulas-list/'
response = requests.get(url)
content = response.content

soup = BeautifulSoup(content, 'html.parser')

table = soup.find('table', class_='table table-bordered')
rows = table.find_all("tr")
for row in rows:
    data = row.find_all("td")
    if data:
        deg = data[0].text.strip(),data[1].text.strip(),data[2].text.strip(),data[3].text.strip(),data[4].text.strip(),data[5].text.strip(),data[6].text.strip(),data[7].text.strip(),data[8].text.strip()
        df = pd.DataFrame(deg)
        print(deg)
        



