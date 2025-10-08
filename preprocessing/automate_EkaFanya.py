import pandas as pd
def preprocess_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset berhasil diproses dan disimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_dataset(
        "dataset_raw/dataset_asli.csv",
        "preprocessing/dataset_preprocessing.csv"
    )
