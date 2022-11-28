import numpy as np
import torch
import intel_extension_for_pytorch as ipex
from torch.nn import Sequential, Linear, CrossEntropyLoss
from torch.optim import SGD


def main():
    device = torch.device("xpu")

    linear = Linear(in_features=10, out_features=1, bias=False)
    print(f"Linear Layer dtype: {linear.weight.dtype}")

    model = Sequential(linear).to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001)

    # This line seems to cause the error
    model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)

    x = torch.rand((5, 10), dtype=torch.float32).to(device)
    y = torch.rand((5, 1), dtype=torch.float32).to(device)
    print(f"X dtype: {x.dtype}")
    print(f"Y dtype: {y.dtype}")

    output = model(x)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Complete")


if __name__ == '__main__':
    main()
