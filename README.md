# Monotonic MLP
Implementation of the monotonic multi-layer perceptron, as described in [*Monotonic Networks* (Sill, 1998)]().

It's very simple to import and implement, since in some sense a monotonic network's first layer is the most critical. Here's an example vanilla implementation:

```python
from sillmmlp import MonotonicLinear, MonotonicGroup, MonotonicMax, MonotonicMin

class MonotonicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = MonotonicLinear(nn.ModuleList([MonotonicGroup(1, 100) for i in range(100)]))
        self.m2 = MonotonicMax()
        self.m3 = MonotonicMin()

    def forward(self, x):
        x = self.m1(x)
        x = self.m2(x)
        x = self.m3(x)
        return x
```
