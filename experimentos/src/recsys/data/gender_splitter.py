import os
from typing import Iterable

import pandas as pd


class GenderSplitter:
    """Separa usuários por gênero e salva listas por janela."""

    def __init__(self, users_path: str = "./data/ml-1m/users.dat"):
        self.users_path = users_path
        self.user_genders = self._load_user_genders()

    def _load_user_genders(self) -> pd.Series:
        columns = ["user", "gender", "age", "occupation", "zip"]
        users = pd.read_csv(
            self.users_path,
            sep="::",
            engine="python",
            names=columns,
            usecols=["user", "gender"],
        )
        users["gender"] = users["gender"].str.upper().str.strip()
        return users.set_index("user")["gender"]

    def split_window(self, base_path: str, output_dir: str, window: int) -> None:
        ratings = pd.read_csv(base_path)
        window_users = pd.Series(ratings["user"].unique(), name="user")
        data = pd.DataFrame(
            {
                "user": window_users,
                "gender": window_users.map(self.user_genders),
            }
        )

        missing = int(data["gender"].isna().sum())
        if missing:
            print(f"Aviso: {missing} usuários sem gênero conhecido na janela {window}.")

        for gender_code, folder in {"M": "male", "F": "female"}.items():
            group_users = data[data["gender"] == gender_code]["user"].sort_values()
            gender_dir = os.path.join(output_dir, folder)
            os.makedirs(gender_dir, exist_ok=True)
            out_path = os.path.join(gender_dir, f"window_{window}_{folder}.csv")
            group_users.to_csv(out_path, index=False, header=False)
            print(f"Janela {window}: {folder} -> {len(group_users)} usuários ({out_path})")

    def split_windows(self, windows: Iterable[int], base_template: str, output_template: str) -> None:
        for window in windows:
            base_path = base_template.format(i=window)
            output_dir = output_template.format(i=window)
            self.split_window(base_path, output_dir, window)


if __name__ == "__main__":
    splitter = GenderSplitter()

    splitter.split_windows(
        windows=range(1, 21),
        base_template="./data/windows/test_to_get_regression_train_data_{i}.csv",
        output_template="./data/windows/gender/hybrid/window_{i}/",
    )

    splitter.split_windows(
        windows=range(1, 21),
        base_template="./data/windows/train_to_get_constituent_methods_{i}.csv",
        output_template="./data/windows/gender/constituent/window_{i}/",
    )
