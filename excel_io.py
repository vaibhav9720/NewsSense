import pandas as pd

def read_alerts(file_path):
    df = pd.read_excel(file_path)
    return df

def write_alerts(df, output_path):
    df.to_excel(output_path, index=False)