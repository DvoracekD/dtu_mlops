import click
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 10
    steps = 0

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_losses.append(loss.item())
            
        else:
            loss = running_loss/len(train_set)
            test_losses.append(loss)
            print(f"Training loss: {loss}")
            
    plt.plot(train_losses)
    plt.savefig("training.png")
    torch.save(model.state_dict(), 'trained_model.pt')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    _, test_set = mnist()

    with torch.no_grad():
        model.eval()
        acc = []
        for images, labels in test_set:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            acc.append(accuracy.item())
        print(f'Accuracy: {torch.tensor(acc).mean()*100}%')
    model.train()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
