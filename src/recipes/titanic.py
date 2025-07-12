"""
Titanic Dataset Recipe
----------------------
Loads and preprocesses the Titanic dataset.
Source: https://www.kaggle.com/c/titanic/data (via seaborn)

Steps:
- Loads from seaborn (no API key needed)
- Basic cleaning and transformation
- Returns a ready-to-use pandas DataFrame
"""

import seaborn as sns


def load_titanic_data():
    """Loads and cleans the Titanic dataset."""

    # Load dataset from seaborn
    df = sns.load_dataset("titanic")

    # Display initial structure the dataset
    print("==Initial dataset structure==")
    print(df.shape, "\n")

    # Drop columns with too many missing values or redundant info
    df = df.drop(columns=["deck", "embark_town", "alive"])

    # Drop rows with critical missing values
    df = df.dropna(subset=["embarked", "age"])

    # Fill missing values in 'embarked' with the most common port

    # First, detect the most common value (mode) in the 'embarked' column
    most_common_embarked = df["embarked"].mode()[0]
    print(f"Most common embarked value: {most_common_embarked} \n")

    # Then, fill missing values with the most common value
    df["embarked"] = df["embarked"].fillna(most_common_embarked)

    # Convert categorical columns to category dtype
    cat_cols = ["sex", "class", "embarked", "who", "adult_male", "alone"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # Display structure of the dataset after our prepriocessing
    print("==Final dataset structure==")
    print(df.shape, "\n")

    return df


# Example usage
if __name__ == "__main__":
    df = load_titanic_data()
    print(df.head())
