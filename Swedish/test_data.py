import os

if __name__ == "__main__":
    cnt = 0
    print("Swedish")
    for file in os.scandir("./train_processed"):
        print(f"\t{file.name}", end=": ")
        print(len(os.listdir(file.path)))
