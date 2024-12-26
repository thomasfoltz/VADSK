import os
import re
import csv

def extract_data(log_file):
    data = {}
    with open(log_file, 'r') as f:
        log_content = f.read()

    train_loss_pattern = r"Average Train Loss:\s+(\d+\.\d+)"
    val_loss_pattern = r"Average Val Loss:\s+(\d+\.\d+)"
    accuracy_pattern = r"Accuracy:\s+(\d+\.\d+)"
    precision_pattern = r"Precision:\s+(\d+\.\d+)"
    recall_pattern = r"Recall:\s+(\d+\.\d+)"
    roc_auc_pattern = r"ROC AUC:\s+(\d+\.\d+)"

    data["Train Loss"] = re.findall(train_loss_pattern, log_content)[0]
    data["Val Loss"] = re.findall(val_loss_pattern, log_content)[0]
    data["Accuracy"] = re.findall(accuracy_pattern, log_content)[0]
    data["Precision"] = re.findall(precision_pattern, log_content)[0]
    data["Recall"] = re.findall(recall_pattern, log_content)[0]
    data["ROC AUC"] = re.findall(roc_auc_pattern, log_content)[0]

    return data

def save_to_csv(results, csv_file):
    header = ["Folder", "Batch Size", "Epochs", "Learning Rate", "Decay",
              "Train Loss", "Val Loss", "Accuracy", "Precision", "Recall", "ROC AUC"]

    sorted_folders = sorted(results.keys())
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for folder in sorted_folders:
            data = results[folder]
            bs, ep, lr, decay = re.findall(r"bs(\d+)_ep(\d+)_lr(\d+\.\d+)_decay(\d+\.\d+)", folder)[0]
            writer.writerow([folder, bs, ep, lr, decay,
                             data['Train Loss'], data['Val Loss'],
                             data['Accuracy'], data['Precision'],
                             data['Recall'], data['ROC AUC']])

if __name__ == "__main__":
    results_dir = "results/SHTech/"
    results = {}

    for folder_name in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder_name)
        if os.path.isdir(folder_path):
            log_file = os.path.join(folder_path, "output.txt")
            if os.path.exists(log_file):
                results[folder_name] = extract_data(log_file)

    save_to_csv(results, "results.csv")