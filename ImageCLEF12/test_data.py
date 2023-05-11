import os
import shutil

if __name__ == "__main__":
    num = []
    print("Reduced ImageCLEF12 train")
    for file in os.scandir("./reduced_train"):
        print(f"\t{file.name}", end=": ")
        print(len(os.listdir(file.path)))
        
    print("Reduced ImageCLEF12 test")
    for file in os.scandir("./reduced_test"):
        print(f"\t{file.name}", end=": ")
        print(len(os.listdir(file.path)))
    
    print("ImageCLEF12 train")
    for file in os.scandir("./train"):
        print(f"\t{file.name}", end=": ")
        print(len(os.listdir(file.path)))
    
    print("ImageCLEF12 test")
    for file in os.scandir("./test"):
        print(f"\t{file.name}", end=": ")
        print(len(os.listdir(file.path)))
    # num = num.sort()
    # num.sort()
    # print(num)