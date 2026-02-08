import os
import pandas as pd


def count_window_files():
    windows_dir = "./data/windows"
    output_csv = "./data/windows/window_records_count.csv"

    rows = []

    # Arquivos de treino/teste usados nas execuções
    patterns = [
        "train_to_get_regression_train_data_",
        "test_to_get_regression_train_data_",
        "train_to_get_constituent_methods_",
        "test_to_get_constituent_methods_",
    ]

    for fname in os.listdir(windows_dir):
        # pula subpastas como full/
        full_path = os.path.join(windows_dir, fname)
        if os.path.isdir(full_path):
            # conta também as janelas completas (full/window_X.csv)
            for subfile in os.listdir(full_path):
                subpath = os.path.join(full_path, subfile)
                if subfile.endswith(".csv"):
                    try:
                        df = pd.read_csv(subpath)
                        rows.append({"file": os.path.join("full", subfile), "count": len(df)})
                    except Exception as e:
                        print(f"Erro lendo {subpath}: {e}")
            continue

        if not fname.endswith(".csv"):
            continue

        if any(fname.startswith(p) for p in patterns):
            try:
                df = pd.read_csv(full_path)
                rows.append({"file": fname, "count": len(df)})
            except Exception as e:
                print(f"Erro lendo {full_path}: {e}")

    if not rows:
        print("Nenhum arquivo contado.")
        return

    out_df = pd.DataFrame(rows)
    # tenta extrair número da janela para ordenar
    def extract_window(fname: str) -> int:
        import re
        m = re.search(r"window_(\d+)", fname)
        return int(m.group(1)) if m else 0

    out_df["window_num"] = out_df["file"].apply(extract_window)
    out_df = out_df.sort_values(by=["window_num", "file"]).drop(columns=["window_num"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Contagens salvas em {output_csv}")


if __name__ == "__main__":
    count_window_files()
