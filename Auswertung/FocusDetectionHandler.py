import fnmatch
import typing
import os
from scidatacontainer import Container
from utils import get_datafiles


def get_focus_result(path) -> str:
    if path[-4:] == ".zdc":
        dc = Container(file=path)
    else:
        path = os.path.join(path, "focus.zdc")
        dc = Container(file=path)
    return dc._items['meas/result.json'].data['statusString']

class Focus_Info_Handler():
    def __init__(self, path):
        self.path = path

    def get_focus_result(self) -> str:
        if self.path[-4:] == ".zdc":
            dc = Container(file=self.path)
        else:
            path = os.path.join(self.path, "focus.zdc")
            dc = Container(file=path)
        return dc._items['meas/result.json'].data['statusString']


# class DHM_handler():
#     def __init__(self, path):
#         self.path = path
#     def get_hologram(self, filename=None) -> "holo_get":
#         if filename is None:
#             if self.path[-4:] == ".zdc":
#                 dc = Container(file=self.path)
#             else:
#                 path = os.path.join(self.path, ".zdc")
#                 try:
#                     dc = Container(file=path)
#                 except FileNotFoundError:
#                     raise FileNotFoundError(f"No file at {path} found.")
#         else:
#             path = os.path.join(self.path, filename)
#             try:
#                 dc = Container(file=path)
#             except Exception as e:
#                 print(f"{type(e)}: {e}")
#         return dc._items['meas/image.png']


def main(root):
    path_list_focus = get_datafiles(root=root, subdir=True, ending="focus.zdc")

    infos = []
    non_focus = []
    for i in range(len(path_list_focus)):
        obj = Focus_Info_Handler(path_list_focus[i])

        infos.append(obj.get_focus_result())

        if infos[-1] != "focus":
            non_focus.append(i)

    return non_focus, infos


# was will ich haben?
# - information wie die differenz der bilder wahrgenommen wurde
# focus, ambigious oder non focus
if __name__ == '__main__':
    complete_list = []
    not_focus_list = []
    name_list = []
    info_list = []
    path_name = []
    root_directory = "C:/Users/hanne/Documents/Seafile/Software/merged2_3D/NanoFactorySystem/test/tools"
    path = os.path.join(root_directory, ".test")

    info_list.append("Detection near together")
    for i in range(2):
        if i == 1:
            path = os.path.join(path, "Single detection")
            info_list.append("Single detection")

        names = fnmatch.filter(os.listdir(path), '*[dz]_?')
        name_list.append(names)
        for j in range(len(names)):
            path_focus = os.path.join(path, names[j])
            path_name.append(path_focus)
            not_list, info = main(path_focus)
            not_focus_list.append(not_list)
            complete_list.append(info)
            if not_list != []:
                if len(not_list) == 100:
                    info_list.append(f"{names[j]} - alle einträge")
                else:
                    info_list.append(f"{names[j]} - {len(not_list)} einträge")

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

# ToDo: location is missing