"""
Script to download Titanic dataset.
"""
import os
import pandas as pd
import ssl
import urllib.request
from pathlib import Path


def download_titanic_data():
    """Download Titanic dataset from seaborn (reliable source)."""
    
    # Get data directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    raw_data_path = project_root / 'data' / 'raw'
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Titanic dataset...")
    
    train_path = raw_data_path / 'train.csv'
    
    if train_path.exists():
        print(f"✓ Dataset already exists, skipping download...")
    else:
        try:
            # Try using seaborn's built-in dataset
            print("Attempting to download from seaborn datasets...")
            import seaborn as sns
            df = sns.load_dataset('titanic')
            df.to_csv(train_path, index=False)
            print(f"✓ Downloaded dataset using seaborn")
        except Exception as e:
            print(f"⚠ Could not download via seaborn: {e}")
            
            # Fallback: try direct download with SSL context
            try:
                print("Attempting direct download...")
                # Create SSL context that doesn't verify certificates
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                
                with urllib.request.urlopen(url, context=ssl_context) as response:
                    data = response.read()
                    with open(train_path, 'wb') as f:
                        f.write(data)
                print(f"✓ Downloaded dataset directly")
            except Exception as e2:
                print(f"✗ Error during direct download: {e2}")
                print("\n⚠ Could not download dataset automatically.")
                print("Please download manually from: https://www.kaggle.com/c/titanic/data")
                return False
    
    # Verify downloaded data
    if train_path.exists():
        df = pd.read_csv(train_path)
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Create a test set from the training data for demonstration
        # In a real scenario, you'd have a separate test set
        print("\nCreating test set from training data...")
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        train_df.to_csv(raw_data_path / 'train.csv', index=False)
        test_df.to_csv(raw_data_path / 'test.csv', index=False)
        
        print(f"✓ Train set: {train_df.shape[0]} rows")
        print(f"✓ Test set: {test_df.shape[0]} rows")
    else:
        print("\n✗ Failed to download dataset")
        return False
    
    print("\n✓ Data download complete!")
    return True


if __name__ == '__main__':
    download_titanic_data()
