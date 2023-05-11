import requests
from bs4 import BeautifulSoup
import os
import shutil
from PIL import Image

if __name__ == "__main__":
    if not os.path.isdir("train_processed"):
        os.mkdir("train_processed")
    response = requests.get('https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/')
    soup = BeautifulSoup(response.text, "html.parser")
    class_id = {}
    for item in soup.select("div#content ul li a"):
        tmp = item.contents[0].split(".")
        id = tmp[0].strip()
        name = tmp[1].strip()
        class_id[f"leaf{id}"] = name

    for dir in os.scandir("train"):
        if not os.path.isdir(f"train_processed/{class_id[dir.name]}"):
            os.mkdir(f"train_processed/{class_id[dir.name]}")

        for img in os.scandir(dir.path):
            inimg = Image.open(img.path)
            outimg = inimg.convert("RGB")

            outimg.save(f"train_processed/{class_id[dir.name]}/{img.name[:-4]}.jpeg", "JPEG", quality=90)

        