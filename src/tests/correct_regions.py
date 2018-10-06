from xml.etree import ElementTree

from region import Region


def load_vatic_regions(path):
    """Parse a VATIC output file to the list of lists consisting of regions for each time step.

    :param path: path of xml file
    :return: list of lists consisting of regions for each time step
    """
    max_polygon = 10000000000000
    tree = ElementTree.parse(path)
    root = tree.getroot()
    all_right_measures = [[] for i in range(250)]
    for child in root.findall('object'):
        print(child.tag)
        for act_count, subchild in enumerate(child):
            if subchild.tag != 'polygon':
                continue
            if act_count > max_polygon:
                raise RuntimeWarning("File has to many objects")

            polygon = list(subchild)
            t = int(polygon[0].text)
            down_lef_point = (int(polygon[1].find("x").text), int(polygon[1].find("y").text))
            up_right_point = (int(polygon[3].find("x").text), int(polygon[3].find("y").text))
            w = up_right_point[0]-down_lef_point[0]
            h = up_right_point[1]-down_lef_point[1]
            measure = Region(down_lef_point[0], down_lef_point[1], w, h)

            if len(all_right_measures) < t+1:
                _resize_list(all_right_measures, t + 1)
            all_right_measures[t].append(measure)
    for t, t_measure_list in enumerate(all_right_measures):
        print(len(t_measure_list), "|t=", t)
    print("Finish")
    return all_right_measures


def _resize_list(all_right_measures, given_size):
    list_len = len(all_right_measures)
    if list_len < given_size:
        all_right_measures.extend([] for i in range(given_size-list_len))
    return all_right_measures
