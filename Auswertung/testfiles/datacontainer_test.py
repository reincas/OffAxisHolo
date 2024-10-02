import os
from scidatacontainer import Container

"""
Testing of reusing/ updating one scidatacontainer again.

1.  load old Container 
    dc = Container(file = path)
2.  get the items of the container
    dictionary = dc.items()
3.  update the dictionary with necessary information
    dictionary.update({"meas/img.png": "Your data"})
4.  create a new container and overwrite with the new data
    dc = Container(items=dictionary)
    dc.write(path)
    
"""


data_path = os.path.join(os.getcwd(), 'testdata')
data_before_path = os.path.join(data_path, 'dhm_DOE_Random_5x5_10µm_before.zdc')
data_after_path = os.path.join(data_path, 'dhm_DOE_Random_5x5_10µm_after.zdc')

dc = Container(file=data_before_path)
# dc.content['complete'] = False
dc_item_dict = dc.items()

for i in range(5):
    # dc._items[f"data/{i}.txt"] = f"das ist ein test für {i}"
    dc_item_dict.update(
        {f"data/{i}_test.txt": f"das ist ein test für {i}"}
    )

# dc2 = Container(items=dc_dict)
# dc2.write(f"{data_path}/test.zdc")

dc = Container(items=dc_item_dict)
dc.write(data_before_path)
print("stop")



def update_container(container_path, update_dict):
    """
    Method for updating an existing SciDataContainer
    :param container_path: path of the container.zdc
    :param update_dict: data which should be updated
    :return: nothing
    """
    if container_path.endswith('.zdc'):
        dc = Container(file=container_path)
    else:
        container_path = container_path + ".zdc"
        dc = Container(file=container_path)

    dc_dictionary = dc.items()
    dc_dictionary.update(update_dict)
    dc = Container(items=dc_dictionary)
    dc.write(container_path)

