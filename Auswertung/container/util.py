"""
Utilities for working wit SciDataContainer.

update_container()          -   updates a container with given dictionary. No information will be lost. BUT it is possible
                                to override information if the key is the same.
set_container_description() -   sets the description of a container. Description has to be a string.
"""

from scidatacontainer import Container


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


def set_container_description(container_path, description):
    if container_path.endswith('.zdc'):
        dc = Container(file=container_path)
    else:
        container_path = container_path + ".zdc"
        dc = Container(file=container_path)

    dc.meta["description"] = description
    dc.write(container_path)