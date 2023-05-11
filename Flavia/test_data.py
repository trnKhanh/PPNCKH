import os

if __name__ == "__main__":
    cnt = 0
    print("Flavia")
    for file in os.scandir("./train"):
        print(f"\t{file.name}", end=": ")
        print(len(os.listdir(file.path)))
