import os

if __name__ == "__main__":
    cnt = 0
    for dir in os.scandir("train_processed"):
        for img in os.scandir(dir.path):
            cnt = cnt + 1
    print(cnt)