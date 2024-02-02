import datasets
import os

DATASET_NAME = ("cis-lmu/Glot500", "por_Latn")

def download_and_concatenate_data(output_file):
    # Step 4: Load the dataset
    dataset = datasets.load_dataset(*DATASET_NAME)

    # Step 5: Access the data splits
    splits_to_save = ["train", "dev", "test"]
    
    # Concatenate train, dev, and test sets
    concatenated_data = []
    for split in splits_to_save:
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in the dataset.")
        
        concatenated_data.extend(dataset[split]['text'])

    # Step 6: Save the concatenated data to a single file
    with open(output_file, "w", encoding="utf-8") as f:
        for text in concatenated_data:
            f.write(text + "\n")

if __name__ == "__main__":
    output_file = "/Users/admin/d2l-en/pytorch/d2l_practice/llm-pretraining/por_file.txt"
    download_and_concatenate_data(output_file)

    # Print the size of the resulting file
    file_size = os.path.getsize(output_file)
    print(f"Size of the concatenated file: {file_size / (1024 * 1024):.2f} MB")

""" 
#! /Users/admin/miniconda3/envs/huggingface/bin/python

# Step 1: Install the required libraries (if you haven't already)
# !pip install datasets

# Step 2: Import the necessary modules
import datasets
import os

DATASET_NAME = ("allenai/nllb", "eng_Latn-kon_Latn")

def save_data_to_files(splits_to_save, output_directory):
    # Step 4: Load the dataset
    dataset = datasets.load_dataset(*DATASET_NAME)

    # Step 5: Access the data splits
    for split in splits_to_save:
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in the dataset.")

        data = dataset[split]

        # Step 6: Save the data to different files
        output_file = os.path.join(output_directory, f"{DATASET_NAME[1]}_{split}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for example in data:
                if isinstance(example, dict) and "translation" in example:
                    text = example["translation"]
                else:
                    raise ValueError("Example format not recognized. Expected a dict with 'text' key.")
                f.write(text['kon_Latn'] + "\n")   # change the value of the key here according to the language


if __name__ == "__main__":
    # Step 3: Choose a dataset
    # dataset_name = "your_dataset_name_here"  # Replace with the name of the dataset you want to download

    # Step 5 (continued): Choose the data splits to save
    splits_to_save = ["train"]  # Customize based on the splits you want to save

    # Step 6 (continued): Choose the output directory
    output_directory = "/Users/admin/d2l-en/pytorch/d2l_practice/llm-pretraining"  # Replace with the desired output directory

    # Call the function to save the data to files
    save_data_to_files(splits_to_save, output_directory) """