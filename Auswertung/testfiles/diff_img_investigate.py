import fnmatch
import glob
import os
from scidatacontainer import Container
import cv2 as cv

from FocusDetectionHandler import Focus_Info_Handler, main, get_focus_result

if __name__ == "__main__":
    complete_list = []
    not_focus_list = []
    name_list = []
    info_list = []
    path_name = []
    root_directory = "C:/Users/hanne/Documents/Seafile/Nanoproduction_Hannes/Code/NanoFactorySystem/mains/.output/test/20240809_planefit63/planefit/layer"
    save_dir= "C:/Users/hanne/Documents/temporary/Focus_bilder_auswertung"

    subdirs = fnmatch.filter(os.listdir(root_directory), '*')

    for subdir in subdirs:
        path = os.path.join(root_directory, subdir)
        subsubdirs = fnmatch.filter(os.listdir(path), '*')
        subsubdirs = subsubdirs[:-1]

        save_subdir = os.path.join(save_dir, subdir)
        # make root_directory

        if not os.path.exists(save_subdir):
            os.mkdir(save_subdir)

        for subsubdir in subsubdirs:
            sub_path = os.path.join(path, subsubdir)
            dc_subpath = os.path.join(sub_path, 'focus.zdc')

            result = get_focus_result(dc_subpath)
            dc = Container(file=dc_subpath)
            img = dc._items['meas/image_diff.png']

            img_name = f"{subsubdir}_{result}.png"

            saving_dir = os.path.join(save_subdir, img_name)

            cv.imwrite(saving_dir, img.data)  # Save the image
            cv.destroyAllWindows()  # Destroy all windows

'''
    # information of ordering in names
    # complete information about every point in complete list
    # information about the non focus points in not_focus_list

    # already done once
    # with open(os.path.join(root_directory, '.test/evaluation.txt'), 'w') as f:
    #     for line in info_list:
    #         f.write(f"{line}\n")
    content = {
        "containerType": {"name": "FocusDetect", "version": 1.1},
    }
    meta = {
        "title": "Focus Detection Data",
        "description": "Detection of laser focus spot on microscope image.",
        "author": "Hannes Robben"
    }

    items = {
        "content.json": content,
        "meta.json": meta,
        "verzeichnis.json": path_name,
        "information_focus_by_point.json": complete_list,
        "no_focus.txt": '\n'.join(map(str, not_focus_list)),
        # "no_focus.txt": '\n'.join(str(x) for x in not_focus_list), # slow way
        # das muss man machen, wenn man eine liste von int hat - vielleicht besser als dat oder so abspeichern
        "evaluation_overview.txt": '\n'.join(info_list)
    }
    dc = Container(items=items)
    dc.write(f"{os.path.join(root_directory, '.test')}/results1.zdc")
    # print(info_list)
'''