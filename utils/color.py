import numpy as np
import cv2
import pickle
import os

from tqdm import tqdm


def read_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


class Color:
    def __init__(
        self,
        image_size: tuple[int] = (256, 256),
        color_space: str = "rgb",
        pyramid: bool = True,
        pacth_sizes: list[tuple[int]] = [(256, 256), (128, 128), (64, 64)],
        hist_type: str = "joint",
        bins: int = 6,
    ) -> None:
        """Clase que permite extraer características de color de una imagen

        Args:
            image_size (tuple[int], optional): Tamaño de la imagen de entrada. Defaults to (256, 256).
            color_space (str, optional): Espacio de color a utilizar. Puede ser 'lab', 'hsv' o 'rgb'. Defaults to 'rgb'.
            pyramid (bool, optional): Usar pirámide de histogramas. Defaults to True.
            pacth_sizes (list[tuple[int]], optional): Tamaño de los bloques de la pirámide. Defaults to [(256, 256), (128, 128), (64, 64)].
            hist_type (str, optional): Tipo de histograma a calcular. Puede ser 'concat' o 'joint'. Defaults to 'joint'.
            bins (int, optional): Número de bins para el histograma. Defaults
            to 6.

        Raises:
            ValueError: Si el espacio de color no es 'lab', 'hsv' o 'rgb'
            ValueError: Si el tipo de histograma no es 'concat' o 'joint'

        """
        self.image_size = image_size
        self.color_space = color_space
        self.pyramid = pyramid
        self.pacth_sizes = pacth_sizes
        self.hist_type = hist_type
        self.bins = bins

    def color_hist(self, Im: np.ndarray) -> np.ndarray:
        """Función que calcula el histograma de color conjunto o concatenado de una imagen

        Args:
            Im (numpy.ndarray): Imagen a la que calcular el histograma

        Returns:
            numpy.ndarray: Histograma de color normalizado
        """
        # YOUR CODE HERE
        if self.hist_type == "concat":
            h = []
            for c in range(Im.shape[2]):
                h.append(np.histogram(Im[..., c].flatten(), bins=self.bins)[0])
            h = np.concatenate((h))
        elif self.hist_type == "joint":
            h = np.histogramdd(Im.reshape(-1, 3), bins=self.bins)[0].flatten()
        else:
            raise ValueError(
                f"El tipo de histograma {self.type_h} no es soportado. Los tipos de histogramas soportados son concat y join"
            )

        h = h / np.sum(h)

        return h

    def Color_pyramid(self, Im: np.ndarray) -> np.ndarray:
        """Función que calcula el histograma de color de una imagen por parches

        Args:
            Im (numpy.ndarray): Imagen de entrada

        Returns:
            numpy.ndarray: Histograma de color normalizado
        """
        # YOUR CODE HERE
        H = []
        for size in self.pacth_sizes:
            for i in range(0, Im.shape[0], size[0]):
                for j in range(0, Im.shape[1], size[1]):
                    H.append(
                        self.color_hist(
                            Im[
                                i : i + size[0],
                                j : j + size[1],
                            ]
                        )
                    )

        H = np.concatenate(H)
        return H

    def __call__(
        self,
        Im_list: list[np.ndarray],
        save: str | None = None,
        split: str | None = None,
    ) -> np.ndarray:
        """Función que extrae las características de color de las imágenes de entrenamiento, validación y test

        Args:
            Im_list (list[np.ndarray]): Lista de imágenes a las que extraer las características de color
            save (str | None, optional): Ruta para guardar descriptores. Defaults to None.
            split (str | None, optional): Nombre del subconjunto de imágenes. Defaults
            to None.

        Returns:
            list[np.ndarray]: Descriptores de color de las imágenes de entrenamiento, validación y test
        """
        if save:
            path = os.path.join(
                save,
                f"{split}_color_{self.image_size[0]}_{self.image_size[1]}_{self.color_space}_{'pyramid_' if self.pyramid else ''}{self.size[0]}_{self.size[1]}_{self.hist_type}_{self.bins}.pkl",
            )

            if os.path.exists(path):
                return read_pickle(path)

        color_features = []
        for Im in tqdm(Im_list, desc="Extracting color features"):
            Im = cv2.resize(Im, self.image_size)
            if self.color_space == "lab":
                Im = cv2.cvtColor(Im, cv2.COLOR_RGB2LAB)
            elif self.color_space == "hsv":
                Im = cv2.cvtColor(Im, cv2.COLOR_RGB2HSV)
            elif self.color_space == "rgb":
                pass
            else:
                raise ValueError(
                    f"El espacio de color {self.color_space} no es soportado. Los tipos de color_space disponibles son lab, hsv y rgb."
                )
            if self.pyramid:
                h = self.Color_pyramid(Im)
            else:
                h = self.color_hist(Im)
            color_features.append(h)

        color_features = np.array(color_features)

        if save:
            if not os.path.exists(save):
                os.makedirs(save, exist_ok=True)

            write_pickle(color_features, path)

        return color_features
