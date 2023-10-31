import os
import re
import csv
import numpy as np
import pandas as pd


class NinaProDataProcessor:
    def __init__(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride
        self.dataset = None
        self.processed_data = None
        self.labels = None
        self.domain_idx = None

    def load_dataset(self, dataset_path):
        return pd.read_csv(dataset_path, header=None).values

    def extract_label_from_path(self, dataset_path):
        match = re.search(r"motion(\d+)", dataset_path)
        if match:
            label = match.group(1)
            return int(label) - 1
        else:
            return None

    def extract_domain_index(self, dataset_path):
        match = re.search(r"s(\d+)", dataset_path)
        if match:
            domain_index = match.group(1)
            return int(domain_index) - 1
        else:
            return None

    def sliding_window(self, data):
        # Total number of samples that can be extracted from this file
        num_samples = (data.shape[0] - self.window_size) // self.stride + 1
        # All samples are stored as (numSamples, windowSize, channels)
        samples = np.empty((num_samples, self.window_size, data.shape[1]))
        for i in range(num_samples):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            samples[i] = data[start_idx:end_idx]

        # For ease of storage, the reconstruction dimension is two-dimensional
        samples = samples.reshape((num_samples, self.window_size * data.shape[1]))

        return samples

    def save_processed_data(self, output_path, array):
        with open(output_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(array)

    def process_data(self, dataset_path, output_path):
        # Load dataset
        self.dataset = self.load_dataset(dataset_path)
        # Perform a sliding window on the dataset
        self.processed_data = self.sliding_window(self.dataset)
        # Get labels
        self.labels = np.array(
            [self.extract_label_from_path(dataset_path)] * self.processed_data.shape[0]
        )
        # Splicing samples and labelling
        labeled_samples = np.column_stack((self.processed_data, self.labels))
        # Get domain index
        self.domain_idx = self.extract_domain_index(dataset_path)
        output_path = os.path.join(
            output_path, f"domain_{self.domain_idx}_labeled_samples.csv"
        )
        # Saving of processed data
        self.save_processed_data(output_path=output_path, array=labeled_samples)


# processor = NinaProDataProcessor(window_size=500, stride=100)
# processor.process_data(
#     dataset_path=r"E:\LocalRepository\NinaPro\rawData\DB2_s3\motion2_repeat1.csv",
#     output_path=r"E:\LocalRepository\NinaPro\processed_data",
# )


def main():
    processor = NinaProDataProcessor(window_size=500, stride=100)
    root_path = r"E:\LocalRepository\NinaPro\rawData"
    domain_list = [os.path.join(root_path, folder) for folder in os.listdir(root_path)]
    for folder in domain_list:
        csv_file_list = [
            os.path.join(folder, file)
            for file in os.listdir(folder)
            if file.endswith(".csv")
        ]
        for dataset_path in csv_file_list:
            processor.process_data(
                dataset_path=dataset_path,
                output_path=r"E:\LocalRepository\NinaPro\processed_data",
            )


if __name__ == "__main__":
    main()
