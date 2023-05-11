import os
import shutil

if __name__ == "__main__":
    num = []
    for file in os.scandir("./reduced_train"):
        # print(os.listdir(file.path)[0])
        # if len(os.listdir(file.path)) < 50:
        #     shutil.rmtree(f"./reduced_train/{file.name}")
        #     shutil.rmtree(f"./reduced_test/{file.name}")
        num.append(len(os.listdir(file.path)))
    
    # num = num.sort()
    num.sort()
    print(num)