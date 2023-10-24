from Config import config

print(config["test_image_path"])

# check if config path is correct
import os
print(os.path.isfile(config["test_image_path"]))

# check current working directory
print(os.getcwd())