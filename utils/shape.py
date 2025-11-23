from skimage.feature import hog
import numpy as np
import pickle
import cv2
import os

from tqdm import tqdm


def read_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


class Shape:
    def __init__(
        self,
        image_size: tuple[int] = (224, 224),
        orientations: int = 9,
        pixels_per_cell: tuple[int] = (32, 32),
        cells_per_block: tuple[int] = (3, 3),
    ) -> None:
        """Clase que permite extraer características de forma de una imagen

        Args:
            image_size (tuple[int], optional): Tamaño de la imagen de entrada. Defaults to (256, 256).
            orientations (int, optional): Número de orientaciones para el histograma de gradientes. Defaults to 9.
            pixels_per_cell (tuple[int], optional): Número de píxeles por celda. Defaults to (8, 8).
            cells_per_block (tuple[int], optional): Número de celdas por bloque. Defaults to (3, 3).
        """
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def __call__(
        self,
        Im_list: list[np.ndarray],
        save: str | None = None,
        split: str | None = None,
    ) -> np.ndarray:
        """Función que extrae las características de forma de las imágenes de entrenamiento, validación y test

        Args:
            Im_list (list): Lista de imágenes de entrada
            save (str | None, optional): Ruta para guardar descriptores. Defaults to None.
            split (str | None, optional): Subconjunto de imágenes. Defaults to None.

        Returns:
            numpy.ndarray: Vector de características de forma
        """
        if save:
            path = os.path.join(
                save,
                f"{split}_shape_{self.image_size[0]}_{self.image_size[1]}_{self.orientations}_{self.pixels_per_cell[0]}_{self.pixels_per_cell[1]}_{self.cells_per_block[0]}_{self.cells_per_block[1]}.pkl",
            )
            if os.path.exists(path):
                return read_pickle(path)

        shape_features = []
        for Im in tqdm(Im_list, desc="Extracting shape features"):
            Im = cv2.cvtColor(cv2.resize(Im, self.image_size), cv2.COLOR_RGB2GRAY)
            shape_features.append(
                hog(
                    Im,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                )
            )

        shape_features = np.array(shape_features)

        if save:
            if os.path.exists(save) == False:
                os.makedirs(save, exist_ok=True)

            write_pickle(shape_features, path)

        return shape_features
