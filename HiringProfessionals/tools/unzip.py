import os
import zipfile

def unzip_files_in_directory(directory):
    """Recursively extract all ZIP files found in directory into same-named subdirectories."""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            if zipfile.is_zipfile(file_path):
                extract_path = os.path.join(root, os.path.splitext(file)[0])
                os.makedirs(extract_path, exist_ok=True)

                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                    print(f"Extracted: {file} -> {extract_path}")

if __name__ == "__main__":
    directory = input("Enter the directory path containing ZIP files: ").strip()
    unzip_files_in_directory(directory)
