import os
import subprocess
import sys

BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")


def run_script(name):
    path = os.path.join(SRC_DIR, name)
    result = subprocess.run([sys.executable, path], cwd=SRC_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    run_script("create_dataset.py")
    run_script("preprocess.py")
    run_script("train_model.py")
    run_script("evaluate.py")
    run_script("visualize.py")
    run_script("predict.py")


if __name__ == "__main__":
    main()
