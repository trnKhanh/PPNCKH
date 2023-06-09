from bs4 import BeautifulSoup
import requests
import os
import shutil

if __name__ == "__main__":
    train_dir = "train"
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    response = requests.get("https://flavia.sourceforge.net/")
    soup = BeautifulSoup(response.text, 'html.parser')
    image_id = []
    img_num = []
    for row in soup.select("body  table  tbody  tr:not(:nth-of-type(1))"):
        name = row.select_one('td:nth-of-type(3)').contents[0]

        id = row.select_one("td:nth-of-type(4)").contents[0].split('-')
        image_id.append({
            "name": name,
            "first": int(id[0]),
            "last": int(id[1])
        })

    for item in image_id:
        img_num.append(item["last"] - item["first"] + 1)
        if not os.path.isdir(f'{train_dir}/{item["name"]}'):
            os.mkdir(f'{train_dir}/{item["name"]}')
        
        for id in range(item["first"], item["last"] + 1):
            shutil.copy(f"Leaves/{id}.jpg", f'{train_dir}/{item["name"]}')

