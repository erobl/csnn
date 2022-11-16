import torch

class ProportionPointSetNN(torch.nn.Module):
    def __init__(self, num_features, architecture=[8,4,2]):
        super().__init__()
        rFFlist = [
            # first convolutional layer
            torch.nn.Linear(num_features, architecture[0]),
            torch.nn.ReLU()
        ]

        if len(architecture) > 1:
            for i, (left, right) in enumerate(zip(architecture, architecture[1:])):
                if i != 0:
                    # rFFlist.append(torch.nn.Sigmoid())
                    rFFlist.append(torch.nn.ReLU())

                rFFlist.append(torch.nn.Linear(left, right))
                # rFFlist.append(torch.nn.BatchNorm2d(right))


        self.rFF = torch.nn.Sequential(
            *rFFlist
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X, return_leaf_probs=False):
        leaf_probs = self.rFF(X)

        leaf_probs = self.sigmoid(leaf_probs)

        pred = leaf_probs.mean(axis=1)


        # leaf_probs = (leaf_probs > 0.5).float()

        if return_leaf_probs:
            return pred, leaf_probs
        else:
            return pred
