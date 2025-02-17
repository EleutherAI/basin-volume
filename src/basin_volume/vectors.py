import torch
from abc import ABC, abstractmethod
from typing import Iterator, Optional
from functools import cached_property

class ImplicitVector(ABC):
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.scale_factor = 1.0
    
    @abstractmethod
    def blocks(self) -> Iterator[torch.Tensor]:
        """Returns iterator over blocks of vector components."""
        pass
    
    @cached_property
    def shape(self) -> torch.Size:
        """Returns shape of the vector."""
        return torch.Size([sum(b.numel() for b in self.blocks())])
    
    def dot(self, other: 'ImplicitVector') -> float:
        """Compute dot product with another implicit vector."""
        result = 0.0
        for b1, b2 in zip(self.blocks(), other.blocks()):
            result += b1 @ b2
        return result * self.scale_factor * other.scale_factor

    def __matmul__(self, other: 'ImplicitVector') -> float:
        """Compute matrix product with another implicit vector."""
        return self.dot(other)

class ImplicitParamVector(ImplicitVector):
    def __init__(self, module: torch.nn.Module, block_size: int):
        super().__init__(block_size)
        self.module = module
    
    def blocks(self) -> Iterator[torch.Tensor]:
        for param in self.module.parameters():
            flat_param = param.view(-1)
            for i in range(0, flat_param.numel(), self.block_size):
                yield flat_param[i:i + self.block_size]
    
    def add_(self, other: ImplicitVector) -> None:
        """Add another vector to this parameter vector."""
        mul_factor = other.scale_factor / self.scale_factor
        with torch.no_grad():
            for param_block, other_block in zip(self.blocks(), other.blocks()):
                param_block.add_(other_block, alpha=mul_factor)
    
    def sub_(self, other: ImplicitVector) -> None:
        """Subtract another vector from this parameter vector."""
        mul_factor = other.scale_factor / self.scale_factor
        with torch.no_grad():
            for param_block, other_block in zip(self.blocks(), other.blocks()):
                param_block.sub_(other_block, alpha=mul_factor)

class ImplicitRandomVector(ImplicitVector):
    def __init__(self, seed: int, ref_vector: ImplicitParamVector):
        super().__init__(ref_vector.block_size)
        self.seed = seed
        self.ref_vector = ref_vector
        self.device = ref_vector.module.device
        self.generator = torch.Generator(device=self.device)
    
    def __mul__(self, scalar: float) -> 'ImplicitRandomVector':
        """Lazy scalar multiplication."""
        self.scale_factor *= scalar
        return self
    
    def __rmul__(self, scalar: float) -> 'ImplicitRandomVector':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'ImplicitRandomVector':
        return self.__mul__(1 / scalar)

    def __neg__(self) -> 'ImplicitRandomVector':
        return self.__mul__(-1)
    
    def blocks(self) -> Iterator[torch.Tensor]:
        self.generator.manual_seed(self.seed)
        for param in self.ref_vector.module.parameters():
            flat_param = param.view(-1)
            for i in range(0, flat_param.numel(), self.block_size):
                block_size = min(self.block_size, flat_param.numel() - i)
                random_block = torch.randn(block_size, generator=self.generator, 
                                         device=self.device)
                yield random_block

    @property
    def norm(self) -> float:
        return torch.sqrt(self @ self)
