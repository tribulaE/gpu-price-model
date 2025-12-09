from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split



# Path to the CSV file 
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gpus.csv"



def preprocess(df):
        """
        1) Separate features (X) and target (y)
        2) One hot encode gpu_name into numeric columns
        """


        # X is everthing except the target
        X = df.drop(columns=["used_price"])

        # y is the target I want to predict
        y = df["used_price"] 

        # Convert GPU name from text to numeric dummy columns
        X = pd.get_dummies(X, columns=["gpu_name"], drop_first=True)



        return X, y


def main():

    # Load the dataset into a DataFrame
    df = pd.read_csv(DATA_PATH)

    # Review of data
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    # Preprocessing the features, target and encoding
    X, y = preprocess(df)
    # Review after preprocessingpy
    print("\nFeatures after encoding (first 5 rows):")
    print(X.head())
    print("Feature matrix shape:", X.shape)

    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nPreprocessing complete.")
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

if __name__ == "__main__":
    main()



