from pathlib import Path
import pandas as pd

# Path to the CSV file 
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gpus.csv"


def main():

    # Load the dataset into a DataFrame
    df = pd.read_csv(DATA_PATH)

    # Inspection
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()