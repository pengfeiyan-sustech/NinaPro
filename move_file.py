import os
import shutil

raw_data_path = r'E:\LocalRepository\NinaPro\rawData'
raw_train_data_path = r'E:\LocalRepository\NinaPro\rawTrainData'
raw_test_data_path = r'E:\LocalRepository\NinaPro\rawTestData'

def move_to_test_folder(folder_name):
    source_folder = os.path.join(raw_data_path, folder_name)
    destination_folder = os.path.join(raw_test_data_path, folder_name)
    os.makedirs(destination_folder, exist_ok=True)
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.csv') and ('repeat2' in file_name or 'repeat5' in file_name):
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            shutil.move(source_file, destination_file)
            
def move_to_train_folder(folder_name):
    source_folder = os.path.join(raw_data_path, folder_name)
    destination_folder = os.path.join(raw_train_data_path, folder_name)
    os.makedirs(destination_folder, exist_ok=True)
    for file_name in os.listdir(source_folder):
        if file_name.endswith('.csv'):
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            shutil.move(source_file, destination_file)
            
            

for folder_name in os.listdir(raw_data_path):
    # move_to_test_folder(folder_name)
    move_to_train_folder(folder_name)