import os
import csv

if __name__ == "__main__":
    root_dir = "./Images"

    with open(f"{root_dir}_pred.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "Prediction", "Label", "Path"])
        for pred in os.scandir(root_dir):
            if not os.path.isdir(pred.path):
                continue

            for label in os.scandir(pred.path):
                if not os.path.isdir(label.path):
                    continue
                for img in os.scandir(label.path):
                    writer.writerow([img.name[:-4], pred.name[5:], label.name, img.path])
            