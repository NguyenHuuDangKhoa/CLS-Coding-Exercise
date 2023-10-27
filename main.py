# Custome Modules
from src.data.read_data import read_data
from src.utils.load_yaml_config import load_yaml_config
from src.utils.create_directories import create_timestamped_folder
from src.features.calculate_mutual_information import compute_mi
from src.features.calculate_mutual_information import mutual_information_binary
from src.features.calculate_mutual_information import mutual_information_multiple_discrete
from src.features.calculate_mutual_information import mutual_info_with_entropy
from src.data.process_data import remove_na, remove_duplicates, remove_outliers_iqr_custom, normalize
from src.visualization.visualize import plot_boxplots, get_stats_sum, plot_histograms, plot_pairplots
from src.visualization.visualize import plot_pearson, plot_spearman, plot_kendall, plot_mutual_information_heatmap
# Third-parties
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

        df = remove_na(data=df)
        df = remove_duplicates(data=df)

        base_dir = config.get("question_1").get("report_path")
        new_folder = create_timestamped_folder(base_dir=base_dir)

        get_stats_sum(data=df, path=new_folder)
        df_id = df["id"]
        df = df.drop(columns=["id"])

        plot_histograms(data=df, path=new_folder)
        plot_boxplots(data=df, path=new_folder, name="boxplots_of_each_feature_original.png")

        df = remove_outliers_iqr_custom(data=df)
        df = normalize(data=df)
        logger.info(f"Dataset shape after removing outliers and normalizing: {df.shape}")
        
        plot_boxplots(data=df, path=new_folder, name="boxplots_of_each_columns_in_normalized_dataset_w_outliers_removed.png")
        plot_pairplots(data=df, path=new_folder)
        plot_pearson(data=df, path=new_folder)
        plot_spearman(data=df, path=new_folder)
        plot_kendall(data=df, path=new_folder)
        plot_mutual_information_heatmap(data=df, path=new_folder)
        
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