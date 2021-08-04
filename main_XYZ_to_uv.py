"""
Skrypt do przeliczania współrzędnych terenowych WCS np fotopunktów do współrzędnych kamery CCS i OPC
Oraz do przygotowania danych dla ML.
"""
import numpy as np
import pandas as pd
from math import sqrt
from multiprocessing import Pool, cpu_count
from itertools import repeat

import tools_pix # biblioteka do pobierania danych z projektów pix4D

import cv2
import time
import os
from shutil import copyfile

class WCStoPCS:
    """
    Klasa przelicza współrzedne punktów terenowych z 'world coordinate system' do współrzednych 'pixel coordinate'.
    Współzedne punktów muszą być podane w tym samym układzie co projekt pix4D.

    Klasa przyjmuje jednen zestawu projekt + punkty
    """

    def __init__(self, pix4d_project: str, coords: pd.DataFrame):
        self.coords = coords
        # self.coords = tools_pix.ReadXYZfile(coords)
        self.proj = tools_pix.ReadDataPix(pix4d_project)
        self.camera_param = self.proj.camera_param()

    @staticmethod
    def XYZ_minus_offset(array_XYZ, array_offset):
        """Odejmuje od wspołrzednych terenowych zadany przez pix4D offset"""
        return array_XYZ.sub(array_offset.squeeze())

    def set_photo_aera(self, altitude: int):
        """definicja do obliczenia maksymalnego zakrezu zdjecia, narazie przybliżona wartość P4RTK dla zdjęć z 150m
        Trzeba dodać bardziej interaktywna obsluge pobierajaca wysokosc lotu z zdjec

        https://docs.google.com/spreadsheets/d/1w6TnSUwykx_mAZcW3cbUGKNwob9TLVXvezWzKeJ8jco/edit?ts=5e171438#gid=2106177553
        """
        self.GSD = (altitude * self.camera_param['pixel_size'][0]/1000000) / (self.camera_param['focal_length'][0]/1000)
        Lx = self.GSD * self.camera_param['image_size_in_pixels'][-1]
        Ly = self.GSD * self.camera_param['image_size_in_pixels'][0]

        photo_area_radius = sqrt((Lx/2)**2 + (Ly/2)**2)

        return photo_area_radius

    def R_XYZoffsetT(self, array_cam_position_rotation: pd.DataFrame, XYZoffset: pd.DataFrame, filtr_photo_darius):
        """https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
        Przelicza world coordinate system(wcs) na camera coordinate system(ccs)
         a nastepnie na observed point coordinate(opc)
         XYZ -> X'Y'Z' -> xh,yh"""

        # Przygotowanie danych
        calculation_X_sub_T = array_cam_position_rotation.copy()

        # Odejmuje od współrzednych punktu współrzedne kamery.
        calculation_X_sub_T['camera_position_t'] = calculation_X_sub_T['camera_position_t'].apply(lambda x: XYZoffset - x)

        # Filtruje zdjecia na któych powinny znajdywać się punkty terenowe
        calculation_X_sub_T['point_photo_radius'] = calculation_X_sub_T['camera_position_t'].apply(lambda x: sqrt(x[0][0]**2 + x[0][1]**2))
        calculation_X_sub_T = calculation_X_sub_T[calculation_X_sub_T['point_photo_radius']<filtr_photo_darius].drop('point_photo_radius', axis=1)

        # position X' = (X', Y', Z') in camera coordinate system is given by:
        ## Mnozy dwie kolumny macierzy.
        array_ccs= calculation_X_sub_T.apply(lambda x: np.matmul(x['camera_rotation_R'],(x['camera_position_t'].T)).reshape(3,), axis = 1)
        return array_ccs

    @staticmethod
    def point_opcs(array_ccs: pd.DataFrame):
        """Oblicza xh, yh czyli krok pośredni pomiędzy policzeniem współrzednych uv"""
        array_opcs = pd.DataFrame(columns=['Photo']).set_index('Photo')
        array_opcs['xh'] = array_ccs.apply(lambda x: x[0]/x[-1])
        array_opcs['yh'] = array_ccs.apply(lambda x: x[1]/x[-1])

        # array_opcs = array_ccs.apply(lambda x: np.array((x[0]/x[-1], x[1]/x[-1])))
        return array_opcs

    @staticmethod
    def xhd_yhd_cords_distortion(array_opcs: pd.DataFrame, array_distortion: pd.DataFrame):
        """Do policzenia współzednych zdjęcia z uwględnienem modelu dystorsi w camera coordinate system (xhd, yhd)
        """
        #Zebranie i odfiltorwanie do tablicy wszystkich danych
        all_data = array_opcs.copy()
        all_data = all_data.merge(array_distortion[['radial_distortion', 'tangential_distortion']], left_index=True, right_index=True)
        all_data['radial_distortion'] = all_data['radial_distortion'].apply(lambda x: x.reshape(3,))
        all_data['tangential_distortion'] = all_data['tangential_distortion'].apply(lambda x: x.reshape(2, ))

        #Przygotowanie danych do obliczen
        array = pd.DataFrame(columns=['Photo']).set_index('Photo')

        array['xh'] = all_data['xh']
        array['yh'] = all_data['yh']

        array['R1'] = all_data['radial_distortion'].apply(lambda x: x[0])
        array['R2'] = all_data['radial_distortion'].apply(lambda x: x[1])
        array['R3'] = all_data['radial_distortion'].apply(lambda x: x[2])

        array['T1'] = all_data['tangential_distortion'].apply(lambda x: x[0])
        array['T2'] = all_data['tangential_distortion'].apply(lambda x: x[1])
        del all_data

        # Policzenie homogeneous point
        array['r'] = array['xh']**2 + array['yh']**2

        # Distorted homogeneous point in camera coordinate system. xhd yhd
        dhp_ccs = pd.DataFrame(columns=['Photo']).set_index('Photo')

        dhp_ccs['xh'] = ((1 + array['R1']*array['r'] + array['R2']*array['r']**2 + array['R3']*array['r']**3) * array['xh'] +
                    2*array['T1']*array['xh']*array['yh'] + array['T2']*(array['r']+2*array['xh']**2))

        dhp_ccs['yh'] = ((1 + array['R1']*array['r'] + array['R2']*array['r']**2 + array['R3']*array['r']**3) * array['yh'] +
                    2*array['T2']*array['xh']*array['yh'] + array['T1']*(array['r'] + 2*array['yh']**2))

        return dhp_ccs

    def _CCStoPCS(self, xh_yh: pd.DataFrame, camera_matrix: pd.DataFrame, point_no: str, buffor:int = 0):
        """Oblicza wspolrzedne uv z dystorsja lub bez zaleznie od podanych wspolrzednych xh yh
        uv z modelem dystorsji wynik funkcji: xhd_yhd_cords_distortion
        uv bez modelu dystorsji wynik funkcji: point_opcs

        Korzystac jedynie z CCStoPCS_prep
        """
        camera_matrix_data = xh_yh.copy()
        camera_matrix_data['fl'] = camera_matrix['camera_matrix_K'].apply(lambda x: x[0,0])
        camera_matrix_data['cx'] = camera_matrix['camera_matrix_K'].apply(lambda x: x[0,2])
        camera_matrix_data['cy'] = camera_matrix['camera_matrix_K'].apply(lambda x: x[1,2])

        point_no = str(point_no)

        u_v = pd.DataFrame(columns=['Photo']).set_index('Photo')
        u_v['u'] = xh_yh['xh'] * camera_matrix_data['fl'] + camera_matrix_data['cx']
        u_v['v'] = xh_yh['yh'] * camera_matrix_data['fl'] + camera_matrix_data['cy']
        u_v['nr'] = point_no

        # Odfiltorwanie punktów poza zakresem zdjęcia, ale nie wiem jak dla zdjec bez dystorsji
        photo_size = self.camera_param['image_size_in_pixels']
        b = buffor
        self.u_v = u_v[(0-b < u_v['v']) & (u_v['v'] <photo_size[0]+b) & (0-b< u_v['u']) & (u_v['u'] <photo_size[1]+b)]
        return self.u_v

    def _CCStoPCS_prep(self, array_XYZ_offset,
                       arrays_camera_rotation,
                       photos_area_radius,
                       arrays_radial_tangetial_distortion,
                       camera_matrix,
                       distortion_model: bool = True,
                       buffer:int = 0):
        """Korzystac z uv_coords_multi"""

        point, row = array_XYZ_offset
        point_xyz = np.array(row)
        array_ccs = self.R_XYZoffsetT(arrays_camera_rotation, point_xyz, photos_area_radius)

        # Observed point coordinate(opc) to xh i yh
        array_opcs = self.point_opcs(array_ccs)

        # Dodanie modelu dystorsji.
        if distortion_model:
            xhd_yhd = self.xhd_yhd_cords_distortion(array_opcs, arrays_radial_tangetial_distortion)
            array_opcs = xhd_yhd

        # The pixel coordinate (xd, yd) of the 3D point projection with distortion model
        uv = self._CCStoPCS(array_opcs, camera_matrix, point, buffer)
        return uv

    def uv_coords_multi(self, distortion_model: bool = True, buffer:int = 0):
        """Oblicza wspolrzedne zdjeciowe punktu na zdjeciach niewyownanych i zapisuje do pliku w wynikach"""

        # Odjęcie zadanego przez pixa offsetu od współrzednych punktów
        array_offset = self.proj.array_offset()
        array_XYZ_offset = self.XYZ_minus_offset(self.coords, array_offset)

        # Przygotowanie tabeli macierzy z pliku wyników pixa
        arrays_calibrated_camera_parameters = self.proj.array_calibrated_camera_parameters()
        arrays_camera_rotation = arrays_calibrated_camera_parameters[['camera_position_t', 'camera_rotation_R']]
        arrays_radial_tangetial_distortion = arrays_calibrated_camera_parameters[['radial_distortion', 'tangential_distortion']]
        camera_matrix = arrays_calibrated_camera_parameters[['camera_matrix_K']]

        # average_altitude = -array_XYZ_offset['Z'].mean()
        average_altitude = 140
        photos_area_radius = self.set_photo_aera(average_altitude)

        args = zip(array_XYZ_offset.iterrows(),
                   repeat(arrays_camera_rotation),
                   repeat(photos_area_radius),
                   repeat(arrays_radial_tangetial_distortion),
                   repeat(camera_matrix),
                   repeat(distortion_model),
                   repeat(buffer))

        # Przygotowanie i puszczenie przeliczania wspolrzednych uv w wielu wątkach.
        pool = Pool(max(int(cpu_count()/2), 1))
        returns = pool.starmap(self._CCStoPCS_prep, args)
        pool.close()
        pool.join()

        # Polaczenie danych wynikowych w jeden obiekt
        self.array_uv = pd.DataFrame(pd.concat(returns), columns=['nr', 'u' , 'v'])
        return self.array_uv


    def save_to_csv(self, array: pd.DataFrame, path_to_save: str):
        array.to_csv(path_to_save, index=True)

    #Do poprawy nie działa dla obecnego wzorca
    # def marks_photos(uv_GCP, photo_path, photos_path_save=None, display=False):
    #     """Wyświetla a takżde zapisuje zdjęcia z zaznaczonym fotopunktem"""
    #     for key_GCP, val_photos in uv_GCP.items():
    #         for key_photo, val_uv in val_photos.items():
    #             path = os.path.join(photo_path, key_photo[:8], key_photo)
    #             photo_path_save = os.path.join(photos_path_save, key_photo[:8], key_photo)
    #             val_uv = tuple([int(x) for x in val_uv])
    #             os.path.isfile(path)
    #             try:
    #                 draw_dot_on_photo(path, val_uv, key_GCP, path_save=photo_path_save)#, display=display)
    #             except:
    #                 print('buuu')
    #


def draw_dots_on_photo(foto_path, GCP_marks:tuple, line, display=False, path_save=None):
        """Do oznaczenia fotopunktów na pojedynczym zdjęciu i jego ewentualne wyświetlenie oraz zapisanie"""
        image = cv2.imread(foto_path) # wczytuje zdjęcie do openCV

        font = cv2.FONT_HERSHEY_SIMPLEX # Wybór czcionki
        fontScale = 5 # Rozmiar czcionki
        color = (0, 0, 255) # Kolor tekstu
        thickness = 1 # grubość linii
        for uv, nr in GCP_marks:
            # image = cv2.putText(image, str(nr), uv, font,
            #                     fontScale, color, thickness, cv2.LINE_AA)
            image = cv2.circle(image, uv, radius=1, color=(0, 0, 255), thickness=-1)

        image = cv2.line(image, line[0], line[1], color, thickness)
        image = cv2.line(image, line[1], line[2], color, thickness)
        image = cv2.line(image, line[2], line[3], color, thickness)
        image = cv2.line(image, line[3], line[0], color, thickness)

        if path_save:
            # Zapisuje zdjęcia w wskazanej ścieżce
            try:
                cv2.imwrite(path_save, image)
            except cv2.error as E:
                print(E)
                print(path_save)
                print(os.path.isfile(path_save))
                print(image)


        if display:
            # wyświetla zdjęcie w małej rozdzielczości
            image2 = cv2.resize(image, (1920, 1020))
            window_name = 'Image' # Nazwa okna z wyświetlonym zdjęciem
            cv2.imshow(window_name, image2)
            cv2.waitKey() # oczekiwanie na kliknięcie przycisku w celu przejścia do kolejnego zdjęcia


def draw_line_on_photo(foto_path, GCP_marks: tuple, line, display=False, path_save=None):
    """Do oznaczenia fotopunktów na pojedynczym zdjęciu i jego ewentualne wyświetlenie oraz zapisanie"""
    image = cv2.imread(foto_path)  # wczytuje zdjęcie do openCV

    font = cv2.FONT_HERSHEY_SIMPLEX  # Wybór czcionki
    fontScale = 5  # Rozmiar czcionki
    color = (0, 0, 255)  # Kolor tekstu
    thickness = 1  # grubość linii
    for uv, nr in GCP_marks:
        # image = cv2.putText(image, str(nr), uv, font,
        #                     fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.circle(image, uv, radius=1, color=(0, 0, 255), thickness=-1)

    image = cv2.line(image, line[0], line[1], color, thickness)
    image = cv2.line(image, line[1], line[2], color, thickness)
    image = cv2.line(image, line[2], line[3], color, thickness)
    image = cv2.line(image, line[3], line[0], color, thickness)

    if path_save:
        # Zapisuje zdjęcia w wskazanej ścieżce
        try:
            cv2.imwrite(path_save, image)
        except cv2.error as E:
            print(E)
            print(path_save)
            print(os.path.isfile(path_save))
            print(image)

    if display:
        # wyświetla zdjęcie w małej rozdzielczości
        image2 = cv2.resize(image, (1920, 1020))
        window_name = 'Image'  # Nazwa okna z wyświetlonym zdjęciem
        cv2.imshow(window_name, image2)
        cv2.waitKey()  # oczekiwanie na kliknięcie przycisku w celu przejścia do kolejnego zdjęcia

def draw_line_on_photo_single(foto_path, box, display=True, path_save=None):
    """Do oznaczenia fotopunktów na pojedynczym zdjęciu i jego ewentualne wyświetlenie oraz zapisanie"""
    image = cv2.imread(foto_path)  # wczytuje zdjęcie do openCV

    font = cv2.FONT_HERSHEY_SIMPLEX  # Wybór czcionki
    fontScale = 5  # Rozmiar czcionki
    color = (0, 0, 255)  # Kolor tekstu
    thickness = 1  # grubość linii
    for uv, nr in GCP_marks:
        # image = cv2.putText(image, str(nr), uv, font,
        #                     fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.circle(image, uv, radius=1, color=(0, 0, 255), thickness=-1)

    image = cv2.line(image, line[0], line[1], color, thickness)
    image = cv2.line(image, line[1], line[2], color, thickness)
    image = cv2.line(image, line[2], line[3], color, thickness)
    image = cv2.line(image, line[3], line[0], color, thickness)

    if path_save:
        # Zapisuje zdjęcia w wskazanej ścieżce
        try:
            cv2.imwrite(path_save, image)
        except cv2.error as E:
            print(E)
            print(path_save)
            print(os.path.isfile(path_save))
            print(image)

    if display:
        # wyświetla zdjęcie w małej rozdzielczości
        image2 = cv2.resize(image, (1920, 1020))
        window_name = 'Image'  # Nazwa okna z wyświetlonym zdjęciem
        cv2.imshow(window_name, image2)
        cv2.waitKey()  # oczekiwanie na kliknięcie przycisku w celu przejścia do kolejnego zdjęcia

def get_list_check(uri, file_extension):
    path = uri
    file_list = []
    for r, d, f in os.walk(path):
        for file in f:
            if file_extension in file:
                fileName = os.path.join(r, file)
                file_list.append(fileName)
    return file_list


def test(path_to_proj, photos_path, path_save_photo, file_save_name):
    start = time.time()
    print(start)

    files = get_list_check(path_save_photo, '.csv')
    f = open(file_save_name, 'a+')
    nor = len(files)
    for no, xyz in enumerate(files):
        print(f'done {no*100/nor}%')
        # if not xyz.endswith('1563.csv'):
        #     print(xyz)
        #     continue
        coords = tools_pix.ReadXYZfile(xyz, ',').dataframe_point_XYZ()
        x = WCStoPCS(path_to_proj, coords)
        photo_size = x.camera_param['image_size_in_pixels']

        try:
            uv = x.uv_coords_multi(distortion_model=True, buffer=30)
        except:
            continue
    
        names = set(uv.index.values.tolist())
        for name_photo in names:
            foo = uv.loc[name_photo]
            # x.save_to_csv(foo, x.proj.set_paths_params('_marks_script_dist_cloud_100_0003_0036.csv'))

            # if name_photo != '100_0154_0003.JPG':
            #     continue

            u_max = int(foo['u'].max())
            u_min = int(foo['u'].min())
            v_max = int(foo['v'].max())
            v_min = int(foo['v'].min())

            def box_min_max(cord, max):
                coo = cord
                if coo < 0:
                    coo = 0

                if coo > max:
                    coo = int(max)
                return int(coo)

            u_max = box_min_max(u_max, photo_size[1])
            u_min = box_min_max(u_min, photo_size[1])
            v_max = box_min_max(v_max, photo_size[0])
            v_min = box_min_max(v_min, photo_size[0])

            box_u = u_max - u_min
            box_v = v_max - v_min

            if box_u < 10 or box_v < 10:
                continue

            # line = [(u_min, v_min), (u_max,v_min), (u_max, v_max), (u_min, v_max)]
            # line = [(u_min, v_min), (u_max, v_min), (u_max, v_max), (u_min, v_max)]
            #
            # try:
            #     GCP_marks = list(foo[['u', 'v']].to_records(index=False, column_dtypes=int))
            # except:
            #     continue
            # GCP_marks = list(tuple(x) for x in GCP_marks)
            #
            # GCP_name = foo[['nr']].to_records(index=False)
            # GCP_name = [int(x[0]) for x in GCP_name]
            #
            #
            # spam = zip(GCP_marks, GCP_name)
            # # for _ in spam:
            # #     print(_)


            photo_path = os.path.join(photos_path, name_photo[:-9], name_photo)
            path_save = os.path.join(path_save_photo, name_photo)
            aaaa= os.path.join(path_save_photo, name_photo)
            try:
                copyfile(photo_path, aaaa)
            except Exception as e:
                print(e)
                continue

            # with open(r'C:\SkySnap_Code\skysnap-skrypty\NCBiR\pointcloud\out\final\Photos\dane.csv', 'a+') as f:
            f.write(f'{name_photo}, {u_min}, {v_min}, {box_u}, {box_v}\n')
            # draw_line_on_photo(photo_path, spam, line, display = display, path_save=path_save)
    f.close()
    print(f'Koniec: {time.time() - start}')

def show_mi(path_to_results, file_save_name, save=False, display=True):
    lista = get_list_check(path_to_results, '.JPG')
    cords = pd.read_csv(file_save_name, names = ['photo', 'x','y','bx','by']).set_index('photo')

    for photo in lista:
        name = os.path.split(photo)[-1]
        try:
            boxs = cords.loc[name]
        except:
            continue
        image = cv2.imread(photo)  # wczytuje zdjęcie do openCV

        font = cv2.FONT_HERSHEY_SIMPLEX  # Wybór czcionki
        fontScale = 5  # Rozmiar czcionki
        color = (0, 0, 255)  # Kolor tekstu
        thickness = 3
        print(f'\n{name}')
        try:
            m = boxs.iterrows()
        except:
            print(boxs)
            print(type(boxs))
            continue
        for coords in m:
            wsp = coords[1]
            wsp = [int(x) for x in wsp]
            LT = (wsp[0], wsp[1])
            RD = (wsp[0] + wsp[2], wsp[1] + wsp[3])
            LD = (wsp[0] + wsp[2], wsp[1])
            RT = (wsp[0], wsp[1] + wsp[3])


            print (LT, RD, LD, RT)
            print (image.shape)
            # for nr, uv in enumerate((LT, RD, LD, RT)):
            #     image = cv2.circle(image, uv, radius=1, color=(0, 0, 255), thickness=-1)
            #     image = cv2.putText(image, str(nr), uv, font,
            #                         fontScale, color, thickness, cv2.LINE_AA)

            image = cv2.line(image, LT, LD, color, thickness)
            image = cv2.line(image, LT, RT, color, thickness)
            image = cv2.line(image, RD, LD, color, thickness)
            image = cv2.line(image, RD, RT, color, thickness)

        if display:
            image2 = cv2.resize(image, (1920, 1020))
            window_name = 'Image'  # Nazwa okna z wyświetlonym zdjęciem
            cv2.imshow(window_name, image2)
            cv2.waitKey()

        dir_save,name = os.path.split(photo)
        path_save = os.path.join(dir_save, 'example', name)
        if save:
            # Zapisuje zdjęcia w wskazanej ścieżce
            try:
                cv2.imwrite(path_save, image)
            except cv2.error as E:
                print(E)
                print(path_save)
                print(os.path.isfile(path_save))
                print(image)

def main():
    home_data = os.getcwd()
    path_to_proj = os.path.join(home_data, 'data', 'pix_proj', '7_Z1_N2_PKP_L8_part1.p4d')
    photos_path = os.path.join(home_data, 'data', 'photos')

    path_save_photo = os.path.join(home_data, 'out')
    os.makedirs(path_save_photo, exist_ok=1)

    name = 'labels.csv'
    result_tst_file = os.path.join(path_save_photo, name)

    test(path_to_proj, photos_path, path_save_photo, result_tst_file)

    show_mi(path_save_photo, result_tst_file, save=True)

if __name__ == '__main__':
    main()
    ...



