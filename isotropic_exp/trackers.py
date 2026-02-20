""" Useful classes for computing running statistics. """

import time
from typing import *

import numpy as np
import torch

from tensor_ops import sqrt, detach, transpose, unsqueeze


class MeanTracker:
    """ Barebones mean tracker. """
    def __init__(self, count=0, sum=0):
        """ Initialization values are just provided for convenience but should be left as default. """
        self.count: int = count
        self.sum: float | np.ndarray | torch.Tensor = sum

    def update(self, sum: float | np.ndarray | torch.Tensor, count: int = 1) -> None:
        """ Typical usage: `tracker.update(loss.sum(0), batch_size).` """
        self.count += count
        self.sum += detach(sum)

    def num_samples(self) -> int:
        return self.count

    def mean(self) -> float | np.ndarray | torch.Tensor:
        return self.sum / self.count

    def to(self, device):
        if torch.is_tensor(self.sum):
            self.sum = self.sum.to(device)

    def cat(trackers: Iterable["MeanTracker"], dim: int) -> "MeanTracker":
        """ Concatenate trackers along a given dimension. Assumes tensors. """
        trackers = list(trackers)
        count = trackers[0].count
        assert all(tracker.count == count for tracker in trackers), "All trackers should have the same count."
        return trackers[0].__class__(count=count, sum=torch.cat([tracker.sum for tracker in trackers], dim=dim))

    def stack(trackers: Iterable["MeanTracker"], dim: int) -> "MeanTracker":
        """ Stack trackers along a given dimension. Assumes tensors. """
        trackers = list(trackers)
        count = trackers[0].count
        assert all(tracker.count == count for tracker in trackers), "All trackers should have the same count."
        return trackers[0].__class__(count=count, sum=torch.stack([tracker.sum for tracker in trackers], dim=dim))


class BatchMeanTracker(MeanTracker):
    def __init__(self, axis=0, count=0, sum=0):
        super().__init__(count=count, sum=sum)
        self.axis = axis

    def update(self, x: Union[np.ndarray, torch.Tensor]):
        """ x is (B, *), returned mean will be (*). """
        super().update(x.sum(self.axis), x.shape[self.axis])


class CovarianceTracker:
    """ Computes E[x*y] - E[x]E[y]. """
    def __init__(self):
        self.x_tracker = MeanTracker()
        self.y_tracker = MeanTracker()
        self.xy_tracker = MeanTracker()

    def update(self, xy: float | np.ndarray | torch.Tensor, x: float | np.ndarray | torch.Tensor, y: Optional[float | np.ndarray | torch.Tensor] = None, count: int = 1) -> None:
        """ Typical usage: `tracker.update((x * y).sum(0), x.sum(0), y.sum(0), batch_size).` """
        self.xy_tracker.update(xy, count)
        self.x_tracker.update(x, count)
        if y is not None:
            self.y_tracker.update(y, count)

    def num_samples(self) -> int:
        return self.xy_tracker.num_samples()

    def mean_x(self) -> float | np.ndarray | torch.Tensor:
        return self.x_tracker.mean()

    def mean_y(self) -> float | np.ndarray | torch.Tensor:
        if self.y_tracker.num_samples() > 0:
            return self._tracker.mean()
        else:
            return self.mean_x()

    def mean(self) -> float | np.ndarray | torch.Tensor:
        """ Alias for mean_x for use when x == y. """
        return self.mean_x()

    def mean_xy(self) -> float | np.ndarray | torch.Tensor:
        return self.xy_tracker.mean()

    def covariance(self) -> float | np.ndarray | torch.Tensor:
        return self.mean_xy() - self.mean_x() * self.mean_y()

    def stddev(self):
        """ Returns standard deviation (only meaningful when x == y and for diagonal values). """
        return sqrt(self.covariance())

    def squared_error_bar(self):
        """ Returns squared error bar (only meaningful when x == y and for diagonal values). """
        return self.covariance() / self.num_samples()

    def error_bar(self):
        """ Returns error bar (only meaningful when x == y and for diagonal values). """
        return self.stddev() / np.sqrt(self.num_samples())

    def to(self, device):
        self.x_tracker.to(device)
        self.y_tracker.to(device)
        self.xy_tracker.to(device)

    def cat(trackers: Iterable["CovarianceTracker"], dim: int) -> "CovarianceTracker":
        """ Concatenate trackers along a given dimension. """
        trackers = list(trackers)
        cat_tracker = trackers[0].__class__()
        cat_tracker.x_tracker = MeanTracker.cat([tracker.x_tracker for tracker in trackers], dim)
        if trackers[0].y_tracker.num_samples() > 0:
            cat_tracker.y_tracker = MeanTracker.cat([tracker.y_tracker for tracker in trackers], dim)
        cat_tracker.xy_tracker = MeanTracker.cat([tracker.xy_tracker for tracker in trackers], dim)
        return cat_tracker

    def stack(trackers: Iterable["CovarianceTracker"], dim: int) -> "CovarianceTracker":
        """ Stack trackers along a given dimension. """
        trackers = list(trackers)
        stack_tracker = trackers[0].__class__()
        stack_tracker.x_tracker = MeanTracker.stack([tracker.x_tracker for tracker in trackers], dim)
        if trackers[0].y_tracker.num_samples() > 0:
            stack_tracker.y_tracker = MeanTracker.stack([tracker.y_tracker for tracker in trackers], dim)
        stack_tracker.xy_tracker = MeanTracker.stack([tracker.xy_tracker for tracker in trackers], dim)
        return stack_tracker


class BatchCovarianceTracker(CovarianceTracker):
    def __init__(self, diag_dims=0, batch_dims=1):
        """ Computes x_ijk y_ijl -> z_ikl. diag_dims and batch_dims are the number of dimensions in i and j indices. """
        super().__init__()
        self.diag_dims = diag_dims
        self.batch_dims = batch_dims
        self.diag_shape = None
        self.x_shape = None
        self.y_shape = None

    def check_shape(self, attr, shape):
        prev_shape = getattr(self, attr)
        if prev_shape is None:
            setattr(self, attr, shape)
        elif prev_shape != shape:
            raise ValueError(f"Got {attr} {shape} which did not match stored {prev_shape}")

    def update(self, x: np.ndarray | torch.Tensor, y: Optional[np.ndarray | torch.Tensor] = None):
        """ x is (D..., B..., C...) and y is (D..., B...., C'....), returned covariance will be (D..., C..., C'...). """
        # Verify diag and batch shapes match between two arguments.
        diag_batch_shape = lambda z: z.shape[:self.diag_dims + self.batch_dims]
        if y is not None and diag_batch_shape(y) != diag_batch_shape(x):
            raise ValueError(f"Incompatible diag and batch shapes: {diag_batch_shape(x)} and {diag_batch_shape(y)}")

        # Decode shapes.
        diag_shape = x.shape[:self.diag_dims]
        batch_shape = x.shape[self.diag_dims:self.diag_dims + self.batch_dims]
        x_shape = x.shape[self.diag_dims + self.batch_dims:]
        y_shape = (y if y is not None else x).shape[self.diag_dims + self.batch_dims:]

        # Set the stored shapes, or verify they match with the stored ones.
        self.check_shape("diag_shape", diag_shape)
        self.check_shape("x_shape", x_shape)
        self.check_shape("y_shape", y_shape)

        # Reshape inputs.
        prod = lambda s: np.prod(s, dtype=int)  # dtype needed for empty tuples
        x = x.reshape((prod(diag_shape), prod(batch_shape), prod(x_shape)))  # (D, B, C)
        y = x.reshape((prod(diag_shape), prod(batch_shape), prod(y_shape))) if y is not None else None  # (D, B, C')

        # Slightly uglier implementation to avoid computing twice the mean of x when y is None.
        count = prod(batch_shape)  # B
        super().update(
            xy=transpose(x) @ (y if y is not None else x).conj(),  # (D, C, C')
            x=x.sum(1),  # (D, C)
            y=(y.sum(1) if y is not None else None),  # (D, C')
            count=count,
        )

    def mean_x(self) -> float | np.ndarray | torch.Tensor:
        """ Return shape (D..., C...). """
        return super().mean_x().reshape(self.diag_shape + self.x_shape)

    def mean_y(self) -> float | np.ndarray | torch.Tensor:
        """ Return shape (D..., C'...). """
        return super().mean_y().reshape(self.diag_shape + self.y_shape)

    def mean_xy(self) -> float | np.ndarray | torch.Tensor:
        """ Return shape (D..., C..., C'...). """
        return super().mean_xy().reshape(self.diag_shape + self.x_shape + self.y_shape)

    def covariance(self) -> np.ndarray | torch.Tensor:
        """ Return shape (D..., C..., C'...). """
        mean_x = unsqueeze(self.mean_x(), dim=-1, num=len(self.y_shape))  # (D..., C..., 1...)
        mean_y = unsqueeze(self.mean_y(), dim=self.diag_dims, num=len(self.x_shape))  # (D..., 1..., C'...)
        return self.mean_xy() - mean_x * mean_y.conj()  # (D..., C..., C'...)

    def cat(trackers: Iterable["BatchCovarianceTracker"], dim: int) -> "BatchCovarianceTracker":
        """ Concatenate trackers along a given dimension. """
        raise NotImplementedError("Too complicated to implement as is as saved trackers do not have the shape information (plus one would need to update it in the returned tracker)")
        trackers = list(trackers)
        cat_tracker = CovarianceTracker.cat(trackers, dim)
        for attr in ["diag_dims", "batch_dims", "diag_shape", "x_shape", "y_shape"]:
            value = getattr(trackers[0], attr)
            assert all(getattr(tracker, attr) == value for tracker in trackers), f"All trackers should have the same {attr}."
            setattr(cat_tracker, attr, value)
        return cat_tracker

    def stack(trackers: Iterable["BatchCovarianceTracker"], dim: int) -> "BatchCovarianceTracker":
        """ Stack trackers along a given dimension. """
        raise NotImplementedError("Too complicated to implement as is as saved trackers do not have the shape information (plus one would need to update it in the returned tracker)")
        trackers = list(trackers)
        stack_tracker = CovarianceTracker.stack(trackers, dim)
        for attr in ["diag_dims", "batch_dims", "diag_shape", "x_shape", "y_shape"]:
            value = getattr(trackers[0], attr)
            assert all(getattr(tracker, attr) == value for tracker in trackers), f"All trackers should have the same {attr}."
            setattr(stack_tracker, attr, value)
        return stack_tracker


class TimeTracker:
    """ TimeTracker to track the time of different repeated steps. Example usage:
    ```
    tracker = TimeTracker()
    tracker.switch("dataloading")
    for x in dataset_name():
        tracker.switch("processing")
        f(x)
        tracker.switch("dataloading")
    times = tracker.stop()
    ```
    """
    def __init__(self):
        self.category: str = None  # Current category
        self.t: float = None  # Time at which we entered the current category
        self.times: Dict[str, float] = {}  # category -> sum of durations

    def update(self):
        """ Private method, which should not be called from user code. """
        if self.category is not None:
            self.times[self.category] = self.times.get(self.category, 0) + time.time() - self.t

    def switch(self, category: str) -> None:
        """ Switch to a new category. """
        self.update()
        self.category = category
        self.t = time.time()

    def stop(self) -> Dict[str, float]:
        """ Returns a dictionary of category -> durations (in seconds). """
        self.update()
        self.category = None
        self.t = None
        return self.times

    def pretty_print(self) -> str:
        """ Returns a pretty string of the durations so far. """
        self.update()
        # Sort by decreasing duration.
        times = list(self.times.items())  # list of (category, duration)
        times.sort(key=lambda x: x[1], reverse=True)  # decreasing duration
        return "\n".join(f"- {category}: {time_delta_to_str(duration)}" for category, duration in times)

    def reset(self) -> None:
        """ Provided for convenience, equivalent to creating a new TimeTracker object. """
        self.category = None
        self.t = None
        self.times = {}


def time_delta_to_str(time_in_seconds: float) -> str:
    """ Pretty prints a timedelta. """
    s = []

    milliseconds = int(time_in_seconds * 1000) % 1000
    if milliseconds > 0:
        s.append(f"{milliseconds:03}ms")

    time_in_seconds = int(time_in_seconds)
    num_seconds = time_in_seconds % 60
    if num_seconds > 0:
        s.append(f"{num_seconds:02}s")

    time_in_minutes = time_in_seconds // 60
    num_minutes = time_in_minutes % 60
    if num_minutes > 0:
        s.append(f"{num_minutes:02}m")

    time_in_hours = time_in_minutes // 60
    num_hours = time_in_hours % 24
    if num_hours > 0:
        s.append(f"{num_hours:02}h")

    time_in_days = time_in_hours // 24
    if time_in_days > 0:
        s.append(f"{time_in_days}d")

    return "".join(reversed(s)).lstrip("0")  # Remove leading zeroes, if any.
