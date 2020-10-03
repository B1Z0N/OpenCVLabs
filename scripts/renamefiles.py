# this script can help to deal with mess in files naming
# after running, files from your directory gonna look like this:
# (new_name)1, (new_name)2, ...
# to use it, just run script and set path to desirable directory
# into first input field and new_name into the next input field

import os

path_to_directory = input("Enter path to directory: ") + "/"
new_name = input("Enter new name for files: ")

try:
    for index, file in enumerate(os.listdir(path_to_directory)):
        extension = file.split(".")[1]
        os.rename(
            path_to_directory + file,
            path_to_directory + new_name + str(index) + "." + extension,
        )
except FileNotFoundError:
    print("Got unccorect directory path")
