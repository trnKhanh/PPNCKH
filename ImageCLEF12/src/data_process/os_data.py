import os
import shutil

if __name__ == '__main__':
    train_dir = "./oversampled_train"
    test_dir = "./oversampled_test"

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    for f in os.scandir("./scan_train"):
        if not os.path.isdir(f'{train_dir}/{f.name}'):
            os.mkdir(f'{train_dir}/{f.name}')
        for img in os.scandir(f.path):
            shutil.copy(img.path, f'{train_dir}/{f.name}')
        cnt = 0
        while len(os.listdir(f'{train_dir}/{f.name}')) < 50:
            cnt += 1
            for img in os.scandir(f.path):
                shutil.copy(img.path, f'{train_dir}/{f.name}/{cnt}_{img.name}')
    
    for f in os.scandir("./scan_test"):
        if not os.path.isdir(f'{test_dir}/{f.name}'):
            os.mkdir(f'{test_dir}/{f.name}')
        for img in os.scandir(f.path):
            shutil.copy(img.path, f'{test_dir}/{f.name}')
        
