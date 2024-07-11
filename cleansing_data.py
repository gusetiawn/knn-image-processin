import os

def set_perfect_banana_dataset(data_path, output_path):
    """
    This function restructures the dataset by renaming the files with an incremental
    prefix 'banana(number)' to indicate a perfect dataset. It does not delete any data.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    counter = 1
    for label in ['mentah', 'matang', 'terlalu_matang']:
        label_path = os.path.join(data_path, label)
        output_label_path = os.path.join(output_path, label)
        if not os.path.exists(output_label_path):
            os.makedirs(output_label_path)
        
        for file in os.listdir(label_path):
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                new_filename = f"banana{counter}.jpg"
                source_file_path = os.path.join(label_path, file)
                destination_file_path = os.path.join(output_label_path, new_filename)
                if not os.path.exists(destination_file_path):
                    os.rename(source_file_path, destination_file_path)
                    counter += 1
                else:
                    print(f"File {destination_file_path} already exists. Skipping rename.")

    print("Dataset has been restructured with banana(number) dataset naming without deleting any data.")

# Example usage
data_path = 'data_sample'
output_path = 'perfect_dataset'
set_perfect_banana_dataset(data_path, output_path)