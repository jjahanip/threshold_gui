import tifffile
import napari
import numpy as np
from skimage.util import img_as_ubyte
from skimage.transform import downscale_local_mean
import pandas as pd


def main(biomarker, filename, tablename, downscale_ratio, thres_range):

    table = pd.read_csv(tablename, index_col='ID')
    col = table.loc[:, biomarker].values

    image = img_as_ubyte(tifffile.imread(filename))
    image = downscale_local_mean(image, (downscale_ratio, downscale_ratio))

    centers = []
    for t in thres_range:
        center = table.loc[col > t / 100, ['centroid_y', 'centroid_x']].values // downscale_ratio
        center = np.concatenate((np.ones((center.shape[0], 1)) * t, center), axis=1).astype(int)
        if center.shape[0] != 0:
            centers.extend(center)
    centers = np.array(centers)

    with napari.gui_qt():
        viewer = napari.view_image(image, name='biomarker', colormap='green')
        viewer.add_points(centers, size=[0, 3, 3], n_dimensional=True)


if __name__ == '__main__':

    biomarker = 'GAD67'       # biomarker name from the probability_table (associative table)
    filename = r'E:\jahandar\DashData\TBI\G2_BR#23_HC_14L\final\R1C7.tif'       # path to the biomarker image
    tablename = r'E:\jahandar\DashData\TBI\G2_BR#23_HC_14L\classification_results\probability_table.csv'    # path to the probability table (associative table)
    downscale_ratio = 1
    thres_range = range(87, 97)
    main(biomarker, filename, tablename, downscale_ratio, thres_range)
