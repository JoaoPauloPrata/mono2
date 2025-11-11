import pandas as pd
import matplotlib.pyplot as plt


base_path = "./data/windows/train_to_get_regression_train_data_"

def plot_user_rating_count_distribution(ratings: pd.DataFrame):
    user_rating_counts = ratings['user'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.hist(user_rating_counts, bins=50, color='blue', alpha=0.7)
    plt.title('User Rating Count Distribution')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.yscale('log')  # Use logarithmic scale for better visibility
    plt.grid(True)
    plt.show()


dataframe = pd.read_csv(base_path + "3.csv")
plot_user_rating_count_distribution(dataframe)