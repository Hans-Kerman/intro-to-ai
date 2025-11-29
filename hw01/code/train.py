from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim

from models import PerceptronModel, RegressionModel

"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model: PerceptronModel, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to prebatch based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        "*** YOUR CODE HERE ***"
        fail = -1
        while fail != 0:
            fail = 0
            for batch in dataloader:
                features = batch["x"]
                label = batch["label"]
                if model.get_prebatchion(features) != label:
                    fail += 1
                    model.w += features * label


def train_regression(model: RegressionModel, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to prebatch based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    count: int = 1
    total_loss: float = 1.0
    epoch = 0
    while True:
        count = 0
        total_loss = 0
        epoch += 1

        for batch in dataloader:
            x = batch["x"]
            label = batch["label"]

            optimizer.zero_grad()

            get_loss = regression_loss(y=label, y_pred=model(x))

            get_loss.backward()
            optimizer.step()

            count += 1
            total_loss += get_loss.item()
        
        if epoch == 5000:
            if total_loss / count > 0.02:
                epoch = 0
            else:
                break
            


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
