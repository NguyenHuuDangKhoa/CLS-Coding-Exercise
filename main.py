# Custome Modules
from src.data.read_data import read_data
from src.utils.load_yaml_config import load_yaml_config
from src.features.calculate_mutual_information import compute_mi
from src.features.calculate_mutual_information import mutual_information_binary
from src.features.calculate_mutual_information import mutual_information_multiple_discrete
from src.features.calculate_mutual_information import mutual_info_with_entropy
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif


if __name__ == "__main__":
    # Load configuration from the config.yaml file
    config = load_yaml_config(path="./config.yml")
    # Read data
    df = read_data(path=config.get("data").get("path"),
                chunksize=config.get("data").get("chunksize"),
                logging=config.get("data").get("logging"))
    print(df.head())
    # Calculate Discrete Mutual Information
    if config.get("question") == 1:
        pass
    else:
        mi_funcs = {
            "mutual_information_binary": mutual_information_binary,
            "mutual_information_multiple_discrete": mutual_information_multiple_discrete,
            "mutual_info_with_entropy": mutual_info_with_entropy,
            "mutual_info_score": mutual_info_score,
            "mutual_info_classif": mutual_info_classif,
        }
        mi_results = compute_mi(data=df,
                                mi_func=mi_funcs[config.get("mutual_information").get("function")])