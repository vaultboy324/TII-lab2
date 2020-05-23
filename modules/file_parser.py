from constants import fields

import os


class FileParser:
    _path = ''
    _data_table = []

    @staticmethod
    def _init(fixed_part, file_name):
        FileParser._path = (os.getcwd() + chr(92)
                            + fixed_part + chr(92)
                            + file_name)

    @staticmethod
    def _create_data_table(parsed_list: list):
        for element in parsed_list:
            if element == '\n':
                break
            string_for_parse = element.replace('\n', '')
            new_array = string_for_parse.split(',')

            fields_count = len(fields.ORDER_FIELD_LIST)

            new_row = {}
            for field_number in range(0, fields_count - 1):
                current_field = fields.ORDER_FIELD_LIST[field_number]
                new_row[current_field] = float(new_array[field_number])
            new_row[fields.NAME] = new_array[fields_count - 1]

            FileParser._data_table.append(new_row)

    @staticmethod
    def get_content(fixed_part, file_name):
        FileParser._init(fixed_part, file_name)
        content = open(FileParser._path, 'r').readlines()
        FileParser._create_data_table(content)
        return FileParser._data_table
