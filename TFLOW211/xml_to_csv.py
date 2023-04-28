import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (int(root.find('size')[1].text),
                     int(root.find('size')[0].text),
                     int(member[4][0].text),
                     int(member[4][2].text),
                     int(member[4][1].text),
                     int(member[4][3].text),
                     root.find('filename').text,
                     member[0].text
                     )
            xml_list.append(value)
    column_name = [ 'height','width', 'xmin', 'xmax', 'ymin', 'ymax', 'filename', 'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'PADVRF')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('PADVRF.csv', index=None)
    print('Successfully converted xml to csv.')


main()
