import os

ROOT_PATH = root_path = ""
for path in os.getcwd().split("\\")[:-2]:
    ROOT_PATH += f"{path}/"