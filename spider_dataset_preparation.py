import json
import os
import gdown

def download_spider_dataset():
    # URLs for Spider dataset files
    urls = {
        "train": "https://drive.google.com/uc?export=download&id=1f12Wj7KCj64rOCoIuxl4OubM1i7blvFp",
        "dev": "https://drive.google.com/uc?export=download&id=1Q2IYTCcGz6mDibCX6o8MR4rZ3L7UPKMx"
    }

    os.makedirs("spider_data", exist_ok=True)

    for dataset, url in urls.items():
        output_path = f"spider_data/{dataset}_spider.json"
        print(f"Downloading {dataset} dataset...")
        gdown.download(url, output_path, quiet=False)

def preprocess_spider_data():
    train_file = "spider_data/train_spider.json"
    dev_file = "spider_data/dev_spider.json"
    processed_data = []

    for file in [train_file, dev_file]:
        with open(file, 'r') as f:
            data = json.load(f)
        
        for item in data:
            processed_data.append({
                "nl_query": item["question"],
                "sql_query": item["query"]
            })
    
    print(f"Processed {len(processed_data)} examples")
    return processed_data

if __name__ == "__main__":
    if not os.path.exists("spider_data/train_spider.json"):
        download_spider_dataset()
    data = preprocess_spider_data()
    
    # Save processed data
    with open("processed_spider_data.json", "w") as f:
        json.dump(data, f)

    print("Data preparation complete. Saved to processed_spider_data.json")
