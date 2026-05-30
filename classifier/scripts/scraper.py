import requests
import pandas as pd
from bs4 import BeautifulSoup


url = "https://muhyidin.id/tabel-lengkap-bahasa-sunda-lemes-a-z/"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

response = requests.get(url, headers=headers)

# tables = pd.read_html(response.text)



html = response.text

soup = BeautifulSoup(html, "html.parser")

rows = []

for tr in soup.find_all("tr"):
    cols = [td.get_text(strip=True) for td in tr.find_all("td")]

    if len(cols) == 4:   # important: only valid rows
        rows.append(cols)

df = pd.DataFrame(rows, columns=["loma", "lemes1", "lemes2", "indonesia"])

print(df.head())

df.to_csv("sunda_lemes_clean.csv", index=False)