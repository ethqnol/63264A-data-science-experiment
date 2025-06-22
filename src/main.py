import flax
from flax import nnx

# Test network using flax
class TestCNN(nnx.Module):
    # NCWH (TF/Flax) vs NWHC (PyTorch)
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        
    def __call__(self, x):
        x = self.conv1(x)
        return x