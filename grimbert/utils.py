from typing import List, Tuple
import torch


def batch_index_select(
    input: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    """Batched version of :func:`torch.index_select`.
    Inspired by https://discuss.pytorch.org/t/batched-index-select/9115/8

    :param input: a torch tensor of shape ``(B, *)`` where ``*``
        is any number of additional dimensions.
    :param dim: the dimension in which to index
    :param index: index tensor of shape ``(B, I)``

    :return: a tensor which indexes ``input`` along dimension ``dim``
        using ``index``. This tensor has the same shape as ``input``,
        except in dimension ``dim``, where it has dimension ``I``.
    """
    batch_size = input.shape[0]

    view = [batch_size] + [1 if i != dim else -1 for i in range(1, len(input.shape))]

    expansion = list(input.shape)
    expansion[0] = batch_size
    expansion[dim] = -1

    return torch.gather(input, dim, index.view(view).expand(expansion))


def find_pattern(lst: list, pattern: list) -> List[Tuple[int, int]]:
    """Search all occurrences of pattern in lst.

    :return: a list of pattern coordinates.
    """
    coords = []
    for i in range(len(lst)):
        if lst[i] == pattern[0] and lst[i : i + len(pattern)] == pattern:
            coords.append((i, i + len(pattern)))
    return coords
