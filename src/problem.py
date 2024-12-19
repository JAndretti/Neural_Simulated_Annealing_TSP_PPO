from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

from utils import repeat_to


class Problem(ABC):
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.generator = torch.Generator(device=device)

    def gain(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.cost(s) - self.cost(self.update(s, a))

    def manual_seed(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @abstractmethod
    def cost(self, s: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        pass

    @abstractmethod
    def generate_params(self) -> Dict[str, torch.Tensor]:
        pass

    @property
    def state_encoding(self) -> torch.Tensor:
        pass

    @abstractmethod
    def generate_init_state(self) -> torch.Tensor:
        pass

    def to_state(self, x: torch.Tensor, temp: torch.Tensor):
        return torch.cat([x, self.state_encoding, repeat_to(temp, x)], -1)

    def from_state(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return state[..., : self.x_dim], state[..., self.x_dim : -1], state[..., -1:]


class TSP(Problem):
    x_dim = 1

    def __init__(
        self,
        dim: int = 50,
        n_problems: int = 256,
        device: str = "cpu",
        params: str = {},
    ):
        """Initialize BinPacking.

        Args:
            dim: num items
            n_problems: batch size
            params: {'weight': torch.Tensor}
        """
        super().__init__(device)
        self.dim = dim
        self.n_problems = n_problems
        self.set_params(**params)

    def set_params(self, coords: torch.Tensor = None) -> None:
        """Set params.

        Args:
            coords: [batch size, dim, 2]
        """
        self.coords = coords

    def generate_params(self, mode: str = "train") -> Dict[str, torch.Tensor]:
        """Generate random coordinates in the unit square.

        Returns:
            coords [batch size, num problems, 2]
        """
        if mode == "test":
            self.manual_seed(0)
        coords = torch.rand(
            self.n_problems, self.dim, 2, device=self.device, generator=self.generator
        )
        return {"coords": coords}

    def cost(self, s: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean tour lengths from city permutations

        Args:
            s: [batch size, dim]
        """
        # Edge lengths
        edge_lengths = self.get_edge_lengths_in_tour(s)
        return torch.sum(edge_lengths, -1)

    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Silly city swap for now

        Args:
            s: perm vector [batch size, coords]
            a: cities to swap ([batch size], [batch size])
        """
        return self.two_opt(s[..., 0], a)[..., None]

    def two_opt(self, x: torch.Tensor, a: torch.Tensor):
        """Swap cities a[0] <-> a[1].

        Args:
            s: perm vector [batch size, coords]
            a: cities to swap ([batch size], [batch size])
        """
        # Two-opt moves invert a section of a tour. If we cut a tour into
        # segments a and b then we can choose to invert either a or b. Due
        # to the linear representation of a tour, we choose always to invert
        # the segment that is stored contiguously.
        left = torch.minimum(a[:, 0], a[:, 1])
        right = torch.maximum(a[:, 0], a[:, 1])
        ones = torch.ones((self.n_problems, 1), dtype=torch.long, device=self.device)
        fidx = torch.arange(self.dim, device=self.device) * ones
        # Reversed indices
        offset = left + right - 1
        ridx = torch.arange(0, -self.dim, -1, device=self.device) + offset[:, None]
        # Set flipped section to all True
        flip = torch.ge(fidx, left[:, None]) * torch.lt(fidx, right[:, None])
        # Set indices to replace flipped section with
        idx = (~flip) * fidx + flip * ridx
        # Perform 2-opt move
        return torch.gather(x, 1, idx)

    @property
    def state_encoding(self) -> torch.Tensor:
        return self.coords

    def get_coords(self, s: torch.Tensor) -> torch.Tensor:
        """Get coords from tour permutation."""
        permutation = s[..., None].expand_as(self.coords).long()
        return self.coords.gather(1, permutation)

    def generate_init_x(self) -> torch.Tensor:
        """Generate random permutations of cities."""
        perm = torch.cat(
            [
                torch.randperm(
                    self.dim, device=self.device, generator=self.generator
                ).view(1, -1)
                for _ in range(self.n_problems)
            ],
            dim=0,
        ).to(self.device)
        return perm[..., None]

    def generate_init_state(self) -> torch.Tensor:
        """State encoding has dims

        [state enc] = [batch size, num items, concat]
        """
        perm = self.generate_init_x()
        return torch.cat([perm, self.state_encoding], -1)

    def get_edge_offsets_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Compute vector to right city in tour

        Args:
            s: [batch size, dim]
        Returns:
            [batch size, dim, 2]
        """
        # Gather dataset in order of tour
        d = self.get_coords(s[..., 0])
        d_roll = torch.roll(d, -1, 1)
        # Edge lengths
        return d_roll - d

    def get_edge_lengths_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Compute distance to right city in tour

        Args:
            s: [batch size, dim, 1]
        Returns:
            [batch size, dim]
        """
        # Edge offsets
        offset = self.get_edge_offsets_in_tour(s)
        # Edge lengths
        return torch.sqrt(torch.sum(offset**2, -1))

    def get_neighbors_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Return distances to neighbors in tour.

        Args:
            s: [batch size, dim, 1] vector
        """
        right_distance = self.get_edge_lengths_in_tour(s)
        left_distance = torch.roll(right_distance, 1, 1)
        return torch.stack([right_distance, left_distance], -1)
