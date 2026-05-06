import os
import urllib.request

def download_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, "Telco-Customer-Churn.csv")
    
    if not os.path.exists(out_path):
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, out_path)
        print(f"Downloaded dataset to {out_path}")
    else:
        print(f"Dataset already exists at {out_path}")

if __name__ == "__main__":
    download_data()
