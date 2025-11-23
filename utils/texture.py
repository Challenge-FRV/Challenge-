import pickle
import os
import numpy as np
from tqdm import tqdm

import torch

from sklearn.cluster import KMeans

import cv2

from scipy.io import loadmat


def read_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def write_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


class Texture:
    def __init__(
        self,
        filterbank: np.ndarray | str = "filterbank.mat",
        textons: int = 25,
        random_state: int = 0,
        img_size: tuple[int] = (256, 256),
        pyramid: bool = True,
        pacth_sizes: list[tuple[int]] = [(256, 256), (128, 128), (64, 64)],
    ) -> None:
        """Clase que permite extraer features de textura de una imagen

        Args:
            Im_list (list): Lista de imágenes de entrada RGB
            filterbank (numpy.ndarray | str): Banco de filtros a utilizar
            textons (int, optional): Textones a utilizar. Defaults to 25.
            random_state (int, optional): Semilla para el generador de números aleatorios. Defaults to 0.
            img_size (tuple[int] | None, optional): Tamaño de las imágenes de entrada. Defaults to (256, 256).
            pyramid (bool, optional): Usar pirámide de histogramas. Defaults to True.
            pacth_sizes (tuple[int] | None, optional): Tamaño de los bloques de la pirámide. Defaults to [(256, 256), (128, 128), (64, 64)].

        """
        self.textons = textons
        self.random_state = random_state
        self.img_size = img_size
        self.pyramid = pyramid
        self.pacth_sizes = pacth_sizes

        if type(filterbank) == np.ndarray:
            filterbank = np.array(filterbank)
        elif type(filterbank) == str:
            filterbank = np.array(loadmat(filterbank)["filterbank"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.filterbank = torch.tensor(
            filterbank, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)[:, None]

    def filter_map(self, Im: np.ndarray) -> torch.tensor:
        """Funcion que aplica un banco de filtros a una imagen

        Args:
            Im (numpy.ndarray): Imagen de entrada RGB

        Returns:
            numpy.ndarray: Resultado de aplicar el banco de filtros a la imagen
        """
        if len(Im.shape) == 3:
            Im = cv2.cvtColor(Im, cv2.COLOR_RGB2GRAY)

        Im = torch.tensor(Im, dtype=torch.float32, device=self.device)[None, None]
        convolution = torch.nn.functional.conv2d(
            Im, self.filterbank, padding="same"
        ).permute(0, 2, 3, 1)
        new_im = convolution.to("cpu").numpy()[0]

        return new_im

    def pixel_dataset(self, Im_list: list) -> list[np.ndarray]:
        """Funcion que aplica un banco de filtros a una lista de imagenes y retorna un arreglo con los pixeles de las imagenes filtradas

        Args:
            Im_list (numpy.ndarray): Lista de imágenes de entrada RGB

        Returns:
            numpy.ndarray: Arreglo con los pixeles de las imagenes filtradas
        """
        assert len(Im_list) > 0, "La lista de imágenes debe tener al menos una imagen"

        pbar = tqdm(total=len(Im_list), desc="Filtering images")

        filtered = self.filter_map(Im_list[0])

        pbar.update(1)

        height, width, num_filters = filtered.shape
        pixels = list(filtered.reshape(height * width, num_filters))

        for Im in Im_list[1:]:
            filtered = self.filter_map(Im)

            height, width, num_filters = filtered.shape
            pixels += list(filtered.reshape(height * width, num_filters))
            pbar.update(1)

        return pixels

    def create_dictionary_of_textons(self, Im_list: list) -> None:
        """Funcion que crea un diccionario de textones a partir de una lista de imagenes

        Args:
            Im_list (list): Lista de imagenes de entrada RGB
        """
        pixels = self.pixel_dataset(Im_list)

        self.dictionary_of_textons = KMeans(
            self.textons, random_state=self.random_state
        ).fit(pixels)

    def texture_map(self, Im: np.ndarray) -> np.ndarray:
        """Funcion que mapea una imagen a textones

        Args:
            Im (numpy.ndarray): Imagen de entrada RGB

        Returns:
            numpy.ndarray: Imagen mapeada a textones
        """
        filtered_im = self.filter_map(Im).astype(float)

        height, width, num_filters = filtered_im.shape

        texture_im = self.dictionary_of_textons.predict(
            filtered_im.reshape(height * width, num_filters)
        ).reshape((height, width))

        return texture_im

    def texture_histogram(
        self,
        Im: np.ndarray,
    ) -> np.ndarray:
        """Funcion que calcula el histograma de textones de una imagen

        Args:
            Im (numpy.ndarray): Imagen de entrada RGB

        Returns:
            numpy.ndarray: Histograma de textones normalizado
        """
        texture_map = self.texture_map(Im)
        if self.pyramid:
            histogram = []
            for size in self.pacth_sizes:
                for i in range(0, Im.shape[0], size[0]):
                    for j in range(0, Im.shape[1], size[1]):
                        hist = np.histogram(
                            texture_map[
                                i : i + size[0],
                                j : j + size[1],
                            ],
                            self.textons,
                            range=(0, self.textons - 1),
                        )[0]
                        hist = hist / np.sum(hist)
                        histogram.append(hist)
            histogram = np.concatenate(histogram)
        else:
            histogram = np.histogram(
                texture_map, self.textons, range=(0, self.textons - 1)
            )[0]
            histogram = histogram / np.sum(histogram)
        return histogram

    def __call__(
        self,
        Im_list: list[np.ndarray],
        save: None | str = None,
        split: None | str = None,
    ) -> np.ndarray:
        """Funcion que extrae los features de textura de las imagenes de entrenamiento, validacion y test

        Args:
            Im_list (list): Lista de imagenes de entrada RGB
            save (str, optional): Ruta donde guardar los features. Defaults to None.
            split (str, optional): Nombre del subconjunto de imagenes. Defaults to None.

        Returns:
            numpy.ndarray: Arreglo con los features de textura de las imagenes

        """
        if save:
            path = os.path.join(
                save,
                f"{split}_text_{self.textons}_{self.img_size[0]}_{self.img_size[1]}_{"pyramid_" if self.pyramid else ""}{self.pixels_per_cell[0]}_{self.pixels_per_cell[1]}.pkl",
            )

            if os.path.exists(path):
                return read_pickle(path)

        if self.img_size != None:
            Im_list = [cv2.resize(Im, self.img_size) for Im in Im_list]
        pbar = tqdm(total=len(Im_list), desc="Extracting texture features")
        features = self.texture_histogram(Im_list[0])[None]
        pbar.update(1)
        for Im in Im_list[1:]:
            features = np.concatenate(
                (features, self.texture_histogram(Im)[None]),
                axis=0,
            )
            pbar.update(1)
        pbar.close()

        if save:
            if os.path.exists(save) == False:
                os.makedirs(save, exist_ok=True)

            write_pickle(features, path)

        return features
