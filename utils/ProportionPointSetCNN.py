import torch

class ProportionPointSetNN(torch.nn.Module):
    def __init__(self, num_features, architecture=[8,4,2]):
        super().__init__()
        rFFlist = [
            # first convolutional layer
            torch.nn.Conv2d(1, architecture[0], (1,num_features)),
            # torch.nn.BatchNorm2d(architecture[0]),
            torch.nn.ReLU()
        ]
        if len(architecture) > 1:
            for i, (left, right) in enumerate(zip(architecture, architecture[1:])):
                if i != 0:
                    # rFFlist.append(torch.nn.Sigmoid())
                    rFFlist.append(torch.nn.ReLU())
                rFFlist.append(torch.nn.Conv2d(left, right, (1,1)))
                # rFFlist.append(torch.nn.BatchNorm2d(right))

        self.rFF = torch.nn.Sequential(
            *rFFlist
        )
        self.sigmoid = torch.nn.Sigmoid()
        """
        self.pooling = torch.nn.Sequential(
            # pooling layer
            torch.nn.AvgPool2d(1,Xtrain.shape[2]),
            # torch.nn.MaxPool2d(1,Xtrain.shape[2]),
            torch.nn.Flatten(),
        )
        """

        self.pooling = lambda x: x.mean(axis=2).flatten(-2,-1)

    def forward(self, X, return_leaf_probs=False):
        X = X.unsqueeze(1)

        leaf_probs = self.rFF(X)

        leaf_probs = self.sigmoid(leaf_probs)

        pred = self.pooling(leaf_probs)

        pred = pred.squeeze()
        leaf_probs.squeeze()

        # leaf_probs = (leaf_probs > 0.5).float()

        if return_leaf_probs:
            return pred, leaf_probs
        else:
            return pred
