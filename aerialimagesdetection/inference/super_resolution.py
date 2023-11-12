import argparse
import shutil

def passthrough():
    shutil.copytree("data", "upscaled_data")

if __name__ == "__main__":
    passthrough()