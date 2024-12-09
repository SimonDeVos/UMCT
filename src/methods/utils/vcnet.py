"""
Utils for the VCNet model. Taken as-is from the original VCNet implementation.
"""

# LOAD MODULES
# Third party
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Data
class Dataset_from_matrix(Dataset):
    """
    Creates a torch dataset from a tensor data_matrix with size n * p.

    Parameters:
        data_matrix (torch.Tensor): A tensor containing the data matrix with size n * p. The first column is the treatment, the remaining columns are features, and the last column is the outcome.
    """
    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_data

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the given index.

        Parameters:
            idx (int or torch.Tensor): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features and the outcome.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1])


def get_iter(data_matrix, batch_size, shuffle=True):
    """
    Creates a DataLoader iterator for the given data matrix.

    Parameters:
        data_matrix (torch.Tensor): A tensor containing the data matrix.
        batch_size (int): The number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader: A DataLoader iterator for the given data matrix.
    """
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator


# Eval
def curve(model, test_matrix, t_grid, targetreg=None):
    """
    Evaluates the model on a test matrix and a grid of treatment values.

    Parameters:
        model (nn.Module): The model to evaluate.
        test_matrix (torch.Tensor): A tensor containing the test data matrix.
        t_grid (torch.Tensor): A tensor containing the grid of treatment values.
        targetreg (nn.Module, optional): The targeted regularizer. Defaults to None.

    Returns:
        torch.Tensor: The evaluated curve.
    """
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]

    test_loader = get_iter(test_matrix, batch_size=test_matrix.shape[0], shuffle=False)

    if targetreg is None:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(x, t)
            out = out[1].data.squeeze()
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse
    else:
        for _ in range(n_test):
            for idx, (inputs, y) in enumerate(test_loader):
                t = inputs[:, 0]
                t *= 0
                t += t_grid[0, _]
                x = inputs[:, 1:]
                break
            out = model.forward(x, t)
            tr_out = targetreg(t).data
            g = out[0].data.squeeze()
            out = out[1].data.squeeze() + tr_out / (g + 1e-6)
            out = out.mean()
            t_grid_hat[1, _] = out
        mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data
        return t_grid_hat, mse


# Model
class Truncated_power:
    """
    Represents a truncated power basis function.

    Parameters:
        degree (int): The degree of the basis function.
        knots (torch.Tensor): A tensor containing the knot positions.
    """
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print("Degree should not set to be 0!")
            raise ValueError

        if not isinstance(self.degree, int):
            print("Degree should be int")
            raise ValueError

    def forward(self, x):
        """
        Parameters:
            x (torch.tensor): batch_size * 1
        Returns:
            the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.0
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = self.relu(x - self.knots[_ - self.degree])
                else:
                    out[:, _] = (
                        self.relu(x - self.knots[_ - self.degree - 1])
                    ) ** self.degree

        return out  # bs, num_of_basis


class Dynamic_FC(nn.Module):
    """
    Represents a dynamic fully connected layer.

    Parameters:
        ind (int): The input dimension.
        outd (int): The output dimension.
        degree (int): The degree of the basis function.
        knots (torch.Tensor): A tensor containing the knot positions.
        act (str, optional): The activation function. Defaults to "relu".
        isbias (int, optional): Whether to include a bias term. Defaults to 1.
        islastlayer (int, optional): Whether this is the last layer. Defaults to 0.
    """
    def __init__(self, ind, outd, degree, knots, act="relu", isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis  # num of basis

        self.weight = nn.Parameter(
            torch.rand(self.ind, self.outd, self.d), requires_grad=True
        )

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        """
        Performs a forward pass through the layer.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T  # bs, outd, d

        x_treat_basis = self.spb.forward(x_treat)  # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2)  # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


def comp_grid(y, num_grid):
    """
    Computes the lower and upper indices and the interpolation distance for the given values.

    Parameters:
        y (torch.Tensor): The input values.
        num_grid (int): The number of grid points.

    Returns:
        tuple: A tuple containing the lower indices, upper indices, and interpolation distances.
    """
    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    """
    Represents a density estimation block.

    Parameters:
        num_grid (int): The number of grid points.
        ind (int): The input dimension.
        isbias (int, optional): Whether to include a bias term. Defaults to 1.
    """
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1

        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, t):
        """
        Performs a forward pass through the density block.

        Parameters:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The treatment values.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)

        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(t, self.num_grid)

        L_out = out[x1, L]
        U_out = out[x1, U]

        out = L_out + (U_out - L_out) * inter

        return out


class VCNet_module(nn.Module):
    """
    Represents the VCNet module.

    Parameters:
        cfg_density (list): Configuration for the density estimator.
        num_grid (int): The number of grid points for the density estimator head.
        cfg (list): Configuration for the dynamics network.
        degree (int): The degree of the basis function.
        knots (torch.Tensor): A tensor containing the knot positions.
        binary_outcome (bool): Whether the outcome is binary.
    """
    def __init__(self, cfg_density, num_grid, cfg, degree, knots, binary_outcome):
        super(VCNet_module, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.cfg_density = cfg_density
        self.num_grid = num_grid

        self.cfg = cfg
        self.degree = degree
        self.knots = knots

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(
                    in_features=layer_cfg[0],
                    out_features=layer_cfg[1],
                    bias=layer_cfg[2],
                )
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(
                    nn.Linear(
                        in_features=layer_cfg[0],
                        out_features=layer_cfg[1],
                        bias=layer_cfg[2],
                    )
                )
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == "relu":
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == "tanh":
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == "sigmoid":
                density_blocks.append(nn.Sigmoid())
            else:
                print("No activation")

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(
            self.num_grid, density_hidden_dim, isbias=1
        )

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                last_layer = Dynamic_FC(
                    layer_cfg[0],
                    layer_cfg[1],
                    self.degree,
                    self.knots,
                    act=layer_cfg[3],
                    isbias=layer_cfg[2],
                    islastlayer=1,
                )
            else:
                blocks.append(
                    Dynamic_FC(
                        layer_cfg[0],
                        layer_cfg[1],
                        self.degree,
                        self.knots,
                        act=layer_cfg[3],
                        isbias=layer_cfg[2],
                        islastlayer=0,
                    )
                )
        blocks.append(last_layer)

        if binary_outcome == True:
            blocks.append(nn.Sigmoid())

        self.Q = nn.Sequential(*blocks)

    def forward(self, x, t):
        """
        Performs a forward pass through the VCNet module.

        Parameters:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The treatment values.

        Returns:
            tuple: A tuple containing the density estimation and the dynamics network output.
        """
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        # t_hidden = torch.cat((torch.unsqueeze(t, 1), x), 1)
        g = self.density_estimator_head(hidden, t)
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        """
        Initializes the weights of the module.
        """
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.0)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()


# Targeted regularizer
class TR(nn.Module):
    """
    Represents the targeted regularizer.

    Parameters:
        degree (int): The degree of the basis function.
        knots (torch.Tensor): A tensor containing the knot positions.
    """
    def __init__(self, degree, knots):
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis  # num of basis
        self.weight = nn.Parameter(torch.rand(self.d), requires_grad=True)

    def forward(self, t):
        """
        Performs a forward pass through the targeted regularizer.

        Parameters:
            t (torch.Tensor): The treatment values.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        """
        Initializes the weights of the module.
        """
        # self.weight.data.normal_(0, 0.01)
        self.weight.data.zero_()


# Training
def criterion(out, y, alpha=0.5, epsilon=1e-6):
    """
    Computes the loss for the VCNet module.

    Parameters:
        out (tuple): A tuple containing the density estimation and the dynamics network output.
        y (torch.Tensor): The target values.
        alpha (float, optional): The weight for the density estimation loss. Defaults to 0.5.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: The computed loss.
    """
    return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(
        out[0] + epsilon
    ).mean()


def criterion_TR(out, trg, y, beta=1.0, epsilon=1e-6):
    """
    Computes the loss for the targeted regularizer.

    Parameters:
        out (tuple): A tuple containing the density estimation and the dynamics network output.
        trg (torch.Tensor): The targeted regularizer output.
        y (torch.Tensor): The target values.
        beta (float, optional): The weight for the targeted regularizer loss. Defaults to 1.0.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: The computed loss.
    """
    # out[1] is Q
    # out[0] is g
    return (
        beta
        * (
            (
                y.squeeze()
                - trg.squeeze() / (out[0].squeeze() + epsilon)
                - out[1].squeeze()
            )
            ** 2
        ).mean()
    )