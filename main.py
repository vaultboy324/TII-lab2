import matplotlib.pyplot as plt

from constants import file_names
from constants import fields
from constants import values
from modules.file_parser import FileParser
from modules.knn import KNN

if __name__ == '__main__':
    dataset = FileParser.get_content(file_names.FIXED_ADDRESS_PART, file_names.FULL_DATA)
    k_list = []
    accuracy_list = []
    for k in range(1, 10):
        accuracy_list.append(KNN.get_result(dataset,
                                            fields.ORDER_FIELD_LIST,
                                            True, k)[fields.ACCURACY])
        k_list.append(k)

    plt.subplot(211)
    plt.plot(k_list, accuracy_list, '-bo')
    plt.grid(True)
    plt.xlabel(fields.K)
    plt.ylabel(fields.ACCURACY)
    plt.title('Со взвешиванием')

    k_list = []
    accuracy_list = []
    for k in range(1, 10):
        accuracy_list.append(KNN.get_result(dataset,
                                            fields.ORDER_FIELD_LIST,
                                            False, k)[fields.ACCURACY])
        k_list.append(k)

    plt.subplot(212)
    plt.plot(k_list, accuracy_list, '-ro')
    plt.grid(True)
    plt.xlabel(fields.K)
    plt.ylabel(fields.ACCURACY)
    plt.title('Без взвешивания')

    plt.show()


    # print(dataset)
    # report = KNN.get_result(dataset, fields.ORDER_FIELD_LIST, values.BASE_WEIGHTS, values.unit_k)
    # for field in report:
    #     print(report[field])
    #
    # report = KNN.get_result(dataset, fields.ORDER_FIELD_LIST, values.BASE_WEIGHTS, values.multiple_k)
    # for field in report:
    #     print(report[field])
    #
    # report = KNN.get_result(dataset, fields.ORDER_FIELD_LIST, values.MODIFY_WEIGHTS, values.multiple_k)
    # for field in report:
    #     print(report[field])
