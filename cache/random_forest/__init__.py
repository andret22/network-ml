import os
# return the absolute path this directory is located in.
def get_path():
    return os.path.dirname(os.path.abspath(__file__))