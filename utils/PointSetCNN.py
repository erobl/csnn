import torch

class PointSetCNN(torch.nn.Module):
    def __init__(self, num_features, architecture=[3,3], head_arch=[3,1], dropout_rate=None):
        super().__init__()
        if dropout_rate is None:
            rFFlist = [
                # first convolutional layer
                torch.nn.Conv2d(1, architecture[0], (1,num_features)),
                torch.nn.BatchNorm2d(architecture[0]),
                torch.nn.ReLU()
            ]
        else:
            rFFlist = [
                # first convolutional layer
                torch.nn.Conv2d(1, architecture[0], (1,num_features)),
                torch.nn.ReLU()
            ]

        if len(architecture) > 1:
            for i, (left, right) in enumerate(zip(architecture, architecture[1:])):
                if i != 0:
                    rFFlist.append(torch.nn.ReLU())

                rFFlist.append(torch.nn.Conv2d(left, right, (1,1)))
                if dropout_rate is not None:
                    rFFlist.append(torch.nn.Dropout2d(p=dropout_rate))
                else:
                    rFFlist.append(torch.nn.BatchNorm2d(right))


        self.rFF = torch.nn.Sequential(
            *rFFlist
        )

        self.activation = torch.nn.ReLU()


        if dropout_rate is None:
            HeadList = [
                torch.nn.Linear(architecture[-1],head_arch[0]),
                torch.nn.BatchNorm1d(head_arch[0]),
                torch.nn.ReLU(),
            ]
        else:
            HeadList = [
                torch.nn.Linear(architecture[-1],head_arch[0]),
                torch.nn.ReLU(),
            ]

        for i, (left,right) in enumerate(zip(head_arch, head_arch[1:])):
                if i != 0:
                    HeadList.append(torch.nn.ReLU())
                HeadList.append(torch.nn.Linear(left,right))
                if dropout_rate is None:
                    HeadList.append(torch.nn.BatchNorm1d(right))

        HeadList.append(torch.nn.Sigmoid())
        HeadList.append(torch.nn.Flatten(0))

        self.head = torch.nn.Sequential(
            *HeadList
        )

    def forward(self, X, return_leaf_probs=False):
        X = X.unsqueeze(1)

        leaf_probs = self.rFF(X)

        leaf_probs = self.activation(leaf_probs)

        pred = self.head(leaf_probs.mean(axis=2).flatten(-2,-1))

        if return_leaf_probs:
            return pred, leaf_probs
        else:
            return pred
