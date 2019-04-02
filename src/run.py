from torch import nn, optim, sparse
import torch
import dlc_practical_prologue as prologue


def train_model(model, criterion, optimizer, train_input, train_target, nb_epochs,side, batch_size):
    for i in range(nb_epochs):
        for idx, elem in enumerate(train_input):
            output = model(elem.view(-1, side*side))
            loss = criterion(output, torch.tensor([train_target[idx]]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_nb_errors(model, target_input, target, batch_size):
    nb_errors = 0
    for idx, elem in enumerate(target_input):
        output = model(elem.view(-1, 196))
        winner = torch.argmax(output, dim=1)
        if winner != target[idx]:
            nb_errors += 1
    return nb_errors


def create_deep_model():
    model = nn.Sequential(
        nn.Linear(196, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
    )
    return model


if __name__ == '__main__':
    nb = 1000
    eta = 0.001
    batch_size = 100
    nb_epochs = 25
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)
    side = list(train_input[0,0,0].size())[0]
    model = create_deep_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=eta)
    train_labels_one_hot = prologue.convert_to_one_hot_labels(torch.zeros(10), train_classes[:, 1])
    train_model(model, criterion, optimizer, train_input[:, 0], train_classes[:, 0], nb_epochs, side, batch_size)
    errors = compute_nb_errors(model, test_input[:, 0], test_classes[:, 0], batch_size)
    print(1 - errors/test_input.size(0))



