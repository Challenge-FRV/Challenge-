import os
import pandas as pd
from glob import glob
import numpy as np

class Dataset:
    def __init__(self, root) -> None:
        """Clase que carga un dataset de imágenes y las distribuye en subconjuntos de entrenamiento, validación y test

        Args:
            root (str, optional): Directorio donde se encuentran las imágenes. Defaults to os.path.join("iNaturalist").
            classification_task (str, optional): Tarea de clasificación. Defaults to "anatomical-landmarks".
        """
        self.root = root
        # Leer las categorías desde el archivo CSV de entretenimiento
        self.categories = pd.read_csv(os.path.join(self.root, 'train', 'train.csv'))
        # Leer las categorias desde las columnas del csv excluir la columna 'ID'
        self.categories = [
            {"id": idx, "name": col}
            for idx, col in enumerate(self.categories.columns[1:])
        ]
        self.load_splits()

    def load_splits(self) -> None:
        """Función para cargar las divisiones de entrenamiento, validación y test del dataset."""
        # Cargar los datos desde los archivos CSV
        self.train_data = pd.read_csv(os.path.join(self.root, 'train', 'train.csv'))
        self.val_data = pd.read_csv(os.path.join(self.root, 'valid', 'valid.csv'))
        # Crear listas de imágenes y etiquetas
        train = []
        for index, row in self.train_data.iterrows():
            ID = row['ID']
            label = tuple(row.iloc[1:])
            label = np.argmax(label)
            image_path = os.path.join(self.root, 'train', ID)
            train.append((image_path, label))
        self.train = train
        val = []
        for index, row in self.val_data.iterrows():
            ID = row['ID']
            label = tuple(row.iloc[1:])
            label = np.argmax(label)
            image_path = os.path.join(self.root, 'valid', ID)
            val.append((image_path, label))
        self.valid = val
        
        self.test = sorted(glob(os.path.join(self.root, 'test', '*.jpg')))

if __name__ == "__main__":
    dataset = Dataset()
