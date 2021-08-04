"""
Biblioteka do zaczytywania wyników projektów pix4d mapper.
"""
import numpy as np
import pandas as pd
import os
from pathlib import Path
from colorama import init, Fore, Back, Style

init(autoreset=True)


class ReadDataPix:
    """WCZYTANIE DANYCH i ich przygotowanie zamiana na macierze z kroku 1_initial"""

    def __init__(self, pix4d_project: str):
        self.pix4d_project = Path(pix4d_project)

    list_array_calibrated_camera_parameters = ['camera_matrix_K', 'radial_distortion',
                                                'tangential_distortion', 'camera_position_t',
                                                'camera_rotation_R']

    def set_paths_params(self, extension):
        """ustawienie sciezek do plikow pixa z folderu initial params."""
        path_base = os.path.splitext(self.pix4d_project)[0]
        pix4d_name = os.path.split(path_base)[1]
        path_base = os.path.join(path_base, '1_initial', 'params')
        return os.path.join(path_base, pix4d_name + extension)

    def array_offset(self):
        """tworzy tabele z pliku offset.xyz"""
        path_offset = self.set_paths_params('_offset.xyz')
        array = pd.read_csv(path_offset, header=None, sep=' ', names=list('XYZ'))
        return array

    def array_pmatrix(self):
        """tworzy tabele z pliku pmatrix.xyz"""
        path_pmatrix = self.set_paths_params('_pmatrix.txt')
        array = pd.read_csv(path_pmatrix, header=None, sep=' ')
        array = array.drop(axis=1, labels=13)
        array.set_index(0, inplace=True)
        return array

    def array_calibrated_camera_parameters(self, *array_names: 'name of chosen arrays', only_one=False):
        """Tworzy wszystkie macierze z pliku calibrated_camera_parameters.txt
        W kolejności:
        camera_matrix_K         [3x3]
        radial_distortion       [3x1]
        tangential_distortion   [2x1]
        camera_position_t       [3x1]
        camera_rotation_R       [3x3]
        Ale możesz wybrać macierze, podając nazwy lub wycinek z zmiennej list_array_calibrated_camera_parameters
        Macierze zwraca są typie array ale znajdują się w liście.
        W przypadku macierzy które są identyczne dla wszystkich zdjęć wybranie only_one spowoduje wygenerowanie macierzy
        tylko dla pierwszego zdjęcia.
        """

        path_calibrated_camera_parameters = self.set_paths_params('_calibrated_camera_parameters.txt')

        def read_x_lines_as_array(x):
            # funkcja do czytania danej ilości linii i zamiana na macierze
            return np.array([list(map(float, f.__next__().strip().split(' '))) for _ in range(x)])

        def skip_x_line(x):
            # funkcja do przeskoczenia wierszy
            for _ in range(x): f.__next__()

        def what_to_do(no_skip_line, read_or_skip):
            if read_or_skip:
                arrays.append(read_x_lines_as_array(no_skip_line))
            else:
                skip_x_line(no_skip_line)

        # W przypadku przekazania dodatkowych parametrów z nazwami macierzy ogarnia to poniższy warunek
        array_name_index = [(3, True, 'camera_matrix_K'),
                            (1, True, 'radial_distortion'),
                            (1, True, 'tangential_distortion'),
                            (1, True, 'camera_position_t'),
                            (3, True, 'camera_rotation_R')]
        if array_names:
            if not type(array_names[0]) == str: array_names = [item for item in array_names[0]]
            names_list = ReadDataPix.list_array_calibrated_camera_parameters
            array_name = [1 if name in array_names else 0 for name in names_list]  # spawdza czy bylo o to zapytanie
            if array_name == [0, 0, 0, 0, 0]: raise KeyError('Bledne nazwy wybranych maciezy')
            for what, no in zip(array_name, range(len(names_list))):
                if not what:
                    array_name_index[no] = (array_name_index[no][0], False, array_name_index[no][-1])


        # Otwarcie pliku i przeskoczenie nagłówków.
        f = open(path_calibrated_camera_parameters)
        skip_x_line(8)

        # stworzenie slownika z wybranymi macierzami z calibrated_camera_parameters.txt
        photo_arrays = {}
        while True:
            try:
                photo, *photo_size = next(f).strip().split(' ')  # read_x_lines_as_array(1).tolist()[0]
                arrays = []
                for no, what, name in array_name_index:
                    what_to_do(no, what)
                photo_arrays[photo] = arrays
                if only_one:
                    break
            except StopIteration:
                break
        f.close()
        photo_arrays_as_list = [[x[0]]+list(x[-1]) for x in photo_arrays.items()]
        columns_name = ['photo']+[x[-1] for x in array_name_index if x[1]]
        photo_arrays = pd.DataFrame(photo_arrays_as_list, columns=columns_name).set_index('photo')
        return photo_arrays

    @staticmethod
    def array_asdict_index(array):
        """Zamienia macierz z postaci 'słowo_kluczowe elementy macierzy' na slownik po indexach 'key:array'"""
        array_headers = list(array.columns.values)
        array = array.to_dict(orient='index')
        array = {key: pd.DataFrame(list(value.values()), index=array_headers) for key, value in array.items()}
        return array

    def camera_param(self):
        """pobiera modelowe parametry kamery z pliku _camera.ssk i zwraca w postaci klucz:[float/string]"""
        path_camera_param = self.set_paths_params('_camera.ssk')
        file = open(path_camera_param, 'r')
        params = {'camera_parameters': file.readline()[24:].strip()}

        for record in file:
            if record == 'end camera_parameters\n':
                break
            record = record.split(':')
            x, y = record
            x = x.strip()
            y = y.split()
            try:
                y = [float(val) for val in y]
            except ValueError:
                y = y[0]
            params[x] = y
        return params

class ReadXYZfile:
    def __init__(self, path_XYZ, sep):
        self.path_XYZ = path_XYZ
        self.separator = sep

    def dataframe_point_XYZ(self):
        """tworzy tabele z listy współrzędnych, plik musi zawierać nagłówki,
        funkcja nietestowana i narażona na błędy"""
        array = pd.read_csv(self.path_XYZ, sep=self.separator)
        array.columns = map(str.upper, array.columns)  # nagłowki z wielkiej litery
        array_headers = list(array.columns.values)  # wyciaga naglowki tabeli
        # array.set_index(array_headers.pop(0), inplace=True)  # wybiera pierwsza kolumne jako opis wierszy
        try:
            array = array[list('XYZ')]  # Ustawia dobra kolejnosc punktow
        except KeyError:
            input(Back.RED + 'Bledne naglowki w pliku z punktami, popraw je i naciśnij enter\n' +
                             'Ewentualnie bledny format, plik powinien posiadac spacje jako przerwy' + Back.RESET +
                  Fore.GREEN + 'prawidlowe to: \tNR X Y Z, kolejnosc wspolrzednych XYZ moze byc dowolna' + Fore.RESET)
            self.dataframe_point_XYZ()
        return array.dropna()
