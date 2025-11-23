import cv2
import numpy as np
from tqdm import tqdm

class Features:
    def __init__(self, color_features, shape_features, texture_features) -> None:
        self.color_features = color_features
        self.shape_features = shape_features
        self.texture_features = texture_features
        
    def get_features_and_labels(self, split: list[tuple[str, int]], isTest=False) -> dict[str, np.ndarray | list[int]]:
        if not isTest:
            Im_list = [cv2.imread(path) for path, _ in tqdm(split, desc="Loading images")]
            labels = [label for _, label in tqdm(split, desc="Loading labels")]
    
        else:
            Im_list = [cv2.imread(path) for path in tqdm(split, desc="Loading images")]
        
        

        texture_features = self.texture_features(Im_list)
        color_features = self.color_features(Im_list)
        shape_features = self.shape_features(Im_list)
        

        
        features_and_labels = {
            "color": color_features,
            "shape": shape_features,
            "texture": texture_features,
            "color_shape": np.concatenate((color_features, shape_features), axis=1),
            "color_texture": np.concatenate((color_features, texture_features), axis=1),
            "shape_texture": np.concatenate((shape_features, texture_features), axis=1),
            "color_shape_texture": np.concatenate((color_features, shape_features, texture_features), axis=1),
        }
        if not isTest:
            features_and_labels["labels"] = labels
        return features_and_labels
                
    def __call__(self, dataset) -> tuple[dict[str, np.ndarray | list[int]], dict[str, np.ndarray | list[int]], dict[str, np.ndarray | list[int]]]:
        print("Extracting features from train split")
        train = self.get_features_and_labels(dataset.train)
        print("Extracting features from valid split")
        valid = self.get_features_and_labels(dataset.valid)
        print("Extracting features from test split")
        test = self.get_features_and_labels(dataset.test, isTest=True)
        return train, valid, test