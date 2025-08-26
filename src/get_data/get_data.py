import json
import re
import requests
from bs4 import BeautifulSoup

def fetch_and_parse(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.find_all('p')
    data = []
    for para in content:
        text = para.get_text(strip=True)
        if text:
            data.append(text)
    return '\n'.join(data)

def extract_law_articles(data_str):
    pattern = re.compile(r'第([一二三四五六七八九十零百]+)条.*?(?=\n第|$)', re.DOTALL)
    lawarticles = {}
    for match in pattern.finditer(data_str):
        articlenumber = match.group(1)
        articlecontent = match.group(0).replace('第' + articlenumber + '条', '').strip()
        lawarticles[f"中华人民共和国劳动法 第{articlenumber}条"] = articlecontent
    return json.dumps(lawarticles, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    url = "https://www.gov.cn/banshi/2005-05/25/content_905.htm"
    data_str = fetch_and_parse(url)
    json_str = extract_law_articles(data_str)
    print(json_str)
