import torch

class PointSetNN(torch.nn.Module):
    def __init__(self, num_features, architecture=[8,4,2]):
        super().__init__()
        rFFlist = [
            # first convolutional layer
            torch.nn.Linear(num_features, architecture[0]),
            # torch.nn.BatchNorm1d(architecture[0]),
            # torch.nn.LayerNorm(architecture[0]),
            torch.nn.ReLU()
        ]

        if len(architecture) > 1:
            for i, (left, right) in enumerate(zip(architecture, architecture[1:])):
                if i != 0:
                    rFFlist.append(torch.nn.ReLU())

                rFFlist.append(torch.nn.Linear(left, right))
                # rFFlist.append(torch.nn.BatchNorm1d(right))
                # rFFlist.append(torch.nn.LayerNorm(right))


        self.rFF = torch.nn.Sequential(
            *rFFlist
        )

        self.sigmoid = torch.nn.Sigmoid()



        self.head = torch.nn.Sequential(
            torch.nn.Linear(architecture[-1],1),
            torch.nn.Sigmoid(),
        )

    def forward(self, X, return_leaf_probs=False):
        leaf_probs = self.rFF(X)

        leaf_probs = self.sigmoid(leaf_probs)

        pred = self.head(leaf_probs.mean(axis=1))

        # pred = torch.sigmoid(1000*(leaf_probs.mean(axis=1) - 0.01))

        if return_leaf_probs:
            return pred, leaf_probs
        else:
            return pred
