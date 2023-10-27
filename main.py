# Custome Modules
from src.data.read_data import read_data
from src.utils.load_yaml_config import load_yaml_config
from src.features.calculate_mutual_information import compute_mi
from src.features.calculate_mutual_information import mutual_information_binary
from src.features.calculate_mutual_information import mutual_information_multiple_discrete
from src.features.calculate_mutual_information import mutual_info_with_entropy
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import structlog


logger = structlog.getLogger(__name__)

if __name__ == "__main__":
    # Load configuration from the config.yaml file
    config = load_yaml_config(path="./config.yml")
    
    # Calculate Discrete Mutual Information
    if config.get("current_question") == "question_1":
        df = read_data(path=config.get("question_1").get("data_path"),
                    chunksize=config.get("question_1").get("chunksize"),
                    logging=config.get("question_1").get("logging"),)
        print(df.head())
        #TODO: Implement more
    elif config.get("current_question") == "question_2":
        df = read_data(path=config.get("question_2").get("data_path"),)
        print(df.head())
        mi_funcs = {
            "mutual_information_binary": mutual_information_binary,
            "mutual_information_multiple_discrete": mutual_information_multiple_discrete,
            "mutual_info_with_entropy": mutual_info_with_entropy,
            "mutual_info_score": mutual_info_score,
            "mutual_info_classif": mutual_info_classif,
        }
        mi_results = compute_mi(data=df,
                                mi_func=mi_funcs[config.get("question_2").get("mutual_information_function")])
    else:
        logger.info("Error! Please change the current_question parameter in config.yml to either question_1 or question_2")