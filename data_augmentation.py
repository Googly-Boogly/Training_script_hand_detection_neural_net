from data_handling import run_txt_to_data,count_lines_in_file


def prune_data(data, labels):
    """
    This function will only take out bad data from the total data
    The data looks like this right now a giant list called all_data
    inside all data is lists called the time_steps there will be 13 or 12 lists inside time_steps
    each list inside time_steps is a list called hand_points this will have 21 lists
    each list inside hand_points is called cords_of_point, which is a list with 3 indexes, index 0 is actual hand point, the next two 1,2 are x,y points
    4d array
    data = [... total_amount_of_data_points]
    time_steps = [... 13]
    hand_points = [... 21]
    cord_of_points = [hand_point, x, y]
    :param data: (list) see above
    :param labels:
    :return:
    """
    return_data = []
    return_labels = []
    for individual in range(len(data)-1):

        if len(data[individual]) > 10:
            if labels[individual] == 'KEEP HAND STEADY' or labels[individual] == 'HAND UP' or labels[individual] == 'HAND DOWN':

                return_data.append(data[individual])
                return_labels.append(labels[individual])

    return {'data': return_data, 'labels': return_labels}


def is_list_of_3_ints(lst):
    return isinstance(lst, list) and len(lst) == 3 and all(isinstance(item, int) for item in lst)


def test_data_structure(data):
    assert isinstance(data, list), "Data is not a list"
    assert len(data) in (12, 13), "Data should contain 12 or 13 inner lists"

    for inner_list in data:
        assert isinstance(inner_list, list), "Inner list is not a list"
        assert len(inner_list) == 21, "Inner list should contain 21 innermost lists"
        for innermost_list in inner_list:
            assert is_list_of_3_ints(innermost_list), "Innermost list should contain 3 integers"


def data_file_to_code(data_file):
    return_data = []
    for x in range(count_lines_in_file(filename=data_file)):
        data = run_txt_to_data(data_file, x + 1)
        return_data.append(data)

    return return_data


def test_neural_network_data(data):
    assert isinstance(data, list), "Data is not a list"
    assert len(data) in (12, 13), "Data should contain 12 or 13 inner lists"
    for innerlist in data:
        assert isinstance(innerlist, list), "Inner list is not a list"
        assert len(innerlist) == 42, "Inner list should contain 21 innermost lists"
        for inner_data in innerlist:
            assert isinstance(inner_data, int), "Inner list is not a int"


def training_data_to_neural_network_ready_data(data):
    """
    will take 1 list in data
    This function will change this is iteration 1
    will need a new name after this training fails and training data changes
    :return:
    """


    new_data = []
    for time_step in data:
        new_time_step = []
        for hand_point in time_step:
            new_time_step.append(hand_point[1])
            new_time_step.append(hand_point[2])
        new_data.append(new_time_step)
    # bug fix some data didnt have the same lenght for unknown reasons
    if len(new_data) == 13:
        new_data.append(new_data[12])
    if len(new_data) == 12:
        new_data.append(new_data[11])
        new_data.append(new_data[11])
    if len(new_data) == 11:
        new_data.append(new_data[10])
        new_data.append(new_data[10])
        new_data.append(new_data[10])
    if len(new_data) == 15:
        new_data.pop(14)

    return new_data


def testing_output_from_training_data_to_neural_network_ready_data(data):
    """
    Only reason for this function is some data from training_data_to_neural_network_ready_data has no length gonna weed them out
    :param data:
    :return:
    """
    if len(data[0]) == 0:
        return False

    runner = 0
    total_zeros = 0
    while runner < len(data):
        if len(data[runner]) == 0:
            total_zeros += 1
            data[runner] = data[runner-1]
        runner += 1
    if total_zeros > 5:
        return False
    else:
        return data

def labels_to_final_label_ready_for_neural_network(labels):
    new_labels = []
    for label in labels:
        if label == 'HAND DOWN':
            new_labels.append([1.0, 0, 0])
        elif label == 'HAND UP':
            new_labels.append([0, 1.0, 0])
        else:
            new_labels.append([0, 0, 1.0])
    return new_labels
