import torch
import torch.nn as nn
import numpy as np
from data_augmentation import test_data_structure, data_file_to_code, prune_data, training_data_to_neural_network_ready_data, test_neural_network_data, testing_output_from_training_data_to_neural_network_ready_data, labels_to_final_label_ready_for_neural_network
from data_handling import count_lines_in_file, run_txt_to_data
# Define your sequences (19 sequences of 19 time steps, each with 42 integers)





class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[-1, :, :])  # Take the output from the last time step
        return output


def testing_neural_net(sequence, labels):
    # sequences = [
    #     [[1, 2, 3, ..., 42],  # Replace ... with the remaining integers for each time step
    #      [4, 5, 6, ..., 42],
    #      # Add your remaining time steps here
    #      ],
        # Add your remaining sequences here
    # ]
    # Convert sequences to a numpy array
    sequences = np.array(sequence)

    # Convert sequences to a PyTorch tensor
    sequences = torch.tensor(sequences, dtype=torch.float)
    sequences = sequences.view(-1, 1, 42)

    # Define the model


    # Define the model
    model = MyModel(input_size=42, hidden_size=32, output_size=3)

    # Print model summary
    print(model)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model with your data
    # You need to have your training data and labels (corresponding to the 3 output neurons)
    # Adjust epochs and batch_size as needed
    # labels = torch.randn((19, 3))  # Replace with your actual labels
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(sequences)
        # labels = labels.view(outputs.shape)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')


def run_neural_network():
    data = data_file_to_code('total_data.txt')
    labels = data_file_to_code('total_labels.txt')
    data_dict = prune_data(data, labels)
    end_data = []
    end_labels = []
    # print(len(data_dict['data']))
    for single_data in data_dict['data']:
        try:
            # test_data_structure(single_data)
            temp_data = training_data_to_neural_network_ready_data(single_data)
            testing_output = testing_output_from_training_data_to_neural_network_ready_data(temp_data)
            # print(testing_output)
            if isinstance(testing_output, list):
                end_data.append(testing_output)
                test_neural_network_data(testing_output)
        except Exception as e:
            pass
    temp_labels = labels_to_final_label_ready_for_neural_network(labels)
    labels = torch.tensor(temp_labels, dtype=torch.float)

    testing_neural_net(end_data, labels)


if __name__ == '__main__':
    run_neural_network()

