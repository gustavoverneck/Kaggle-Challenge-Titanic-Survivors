# /src/train.py

import argparse
from config import TRAINING_FILE, OUTPUT_PATH
from model_dispatcher import models
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def run(fold, model):
    # Load data
    df = pd.read_csv(TRAINING_FILE)



    # Save the model
    joblib.dump(clf, os.path.join(OUTPUT_PATH, f"../models/dt_{fold}.bin"))
    return

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--fold", type=int)
    # Read arguments
    args = parser.parse_args()
    run(fold=args.fold)