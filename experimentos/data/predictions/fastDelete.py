import os
import shutil

folders = ["BIAS", "BIASEDMF", "itemKNN", "SVD", "userKNN"]
base_path = os.path.dirname(os.path.abspath(__file__))

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Erro ao deletar {file_path}: {e}")
    else:
        print(f"Pasta não encontrada: {folder_path}")