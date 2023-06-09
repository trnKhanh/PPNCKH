import os
import xml.dom.minidom
import shutil

if __name__ == "__main__":
    train_dir = "../data/scan_train"
    test_dir = "../data/scan_test"

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)
    for file in os.scandir("../FourniApresCLEF2012/data/train"):
        if file.path.endswith(".jpg"):
            id = file.path[:-4]
            information = xml.dom.minidom.parse(f'{id}.xml')
            if information.getElementsByTagName("Type")[0].firstChild.nodeValue == 'photograph':
                continue

            name = information.getElementsByTagName("ClassId")[0].firstChild.nodeValue
            print(name)
            if not os.path.isdir(f'{train_dir}/{name}'):
                os.mkdir(f'{train_dir}/{name}')
            if not os.path.isdir(f'{test_dir}/{name}'):
                os.mkdir(f'{test_dir}/{name}')

            shutil.copy(file.path, f"{train_dir}/{name}")

    for file in os.scandir("../FourniApresCLEF2012/data/test"):
        if file.path.endswith(".jpg"):
            id = file.name[:-4]
            information = xml.dom.minidom.parse(f'../FourniApresCLEF2012/data/testwithgroundtruthxml/{id}.xml')
            if information.getElementsByTagName("Type")[0].firstChild.nodeValue == 'photograph':
                continue
            
            name = information.getElementsByTagName("ClassId")[0].firstChild.nodeValue
            print(name)
            if not os.path.isdir(f'{test_dir}/{name}'):
                os.mkdir(f'{test_dir}/{name}')

            shutil.copy(file.path, f"{test_dir}/{name}")
