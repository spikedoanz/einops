import sys
import keyword
import warnings
from typing import List, Optional, Set, Tuple, Union
import functools
import itertools
import string
import typing
from collections import OrderedDict
from typing import Set, Tuple, List, Dict, Union, Callable, Optional, TypeVar, cast, Any


_ellipsis: str = "…"  # NB, this is a single unicode symbol. String is used as it is not a list, but can be iterated



__author__ = "Alex Rogozhnikov"

_loaded_backends: dict = {}
_type2backend: dict = {}
_debug_importing = False


class AnonymousAxis(object):
    """Important thing: all instances of this class are not equal to each other"""

    def __init__(self, value: str):
        self.value = int(value)
        if self.value <= 1:
            if self.value == 1:
                raise ValueError("No need to create anonymous axis of length 1. Report this as an issue")
            else:
                raise ValueError("Anonymous axis should have positive length, not {}".format(self.value))

    def __repr__(self):
        return "{}-axis".format(str(self.value))


class ParsedExpression:
    """
    non-mutable structure that contains information about one side of expression (e.g. 'b c (h w)')
    and keeps some information important for downstream
    """

    def __init__(self, expression: str, *, allow_underscore: bool = False, allow_duplicates: bool = False):
        self.has_ellipsis: bool = False
        self.has_ellipsis_parenthesized: Optional[bool] = None
        self.identifiers: Set[str] = set()
        # that's axes like 2, 3, 4 or 5. Axes with size 1 are exceptional and replaced with empty composition
        self.has_non_unitary_anonymous_axes: bool = False
        # composition keeps structure of composite axes, see how different corner cases are handled in tests
        self.composition: List[Union[List[str], str]] = []
        if "." in expression:
            if "..." not in expression:
                raise ValueError("Expression may contain dots only inside ellipsis (...)")
            if str.count(expression, "...") != 1 or str.count(expression, ".") != 3:
                raise ValueError(
                    "Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor "
                )
            expression = expression.replace("...", _ellipsis)
            self.has_ellipsis = True

        bracket_group: Optional[List[str]] = None

        def add_axis_name(x):
            if x in self.identifiers:
                if not (allow_underscore and x == "_") and not allow_duplicates:
                    raise ValueError('Indexing expression contains duplicate dimension "{}"'.format(x))
            if x == _ellipsis:
                self.identifiers.add(_ellipsis)
                if bracket_group is None:
                    self.composition.append(_ellipsis)
                    self.has_ellipsis_parenthesized = False
                else:
                    bracket_group.append(_ellipsis)
                    self.has_ellipsis_parenthesized = True
            else:
                is_number = str.isdecimal(x)
                if is_number and int(x) == 1:
                    # handling the case of anonymous axis of length 1
                    if bracket_group is None:
                        self.composition.append([])
                    else:
                        pass  # no need to think about 1s inside parenthesis
                    return
                is_axis_name, reason = self.check_axis_name_return_reason(x, allow_underscore=allow_underscore)
                if not (is_number or is_axis_name):
                    raise ValueError("Invalid axis identifier: {}\n{}".format(x, reason))
                if is_number:
                    x = AnonymousAxis(x)
                self.identifiers.add(x)
                if is_number:
                    self.has_non_unitary_anonymous_axes = True
                if bracket_group is None:
                    self.composition.append([x])
                else:
                    bracket_group.append(x)

        current_identifier = None
        for char in expression:
            if char in "() ":
                if current_identifier is not None:
                    add_axis_name(current_identifier)
                current_identifier = None
                if char == "(":
                    if bracket_group is not None:
                        raise ValueError("Axis composition is one-level (brackets inside brackets not allowed)")
                    bracket_group = []
                elif char == ")":
                    if bracket_group is None:
                        raise ValueError("Brackets are not balanced")
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ["_", _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise ValueError("Unknown character '{}'".format(char))

        if bracket_group is not None:
            raise ValueError('Imbalanced parentheses in expression: "{}"'.format(expression))
        if current_identifier is not None:
            add_axis_name(current_identifier)

    def flat_axes_order(self) -> List:
        result = []
        for composed_axis in self.composition:
            assert isinstance(composed_axis, list), "does not work with ellipsis"
            for axis in composed_axis:
                result.append(axis)
        return result

    def has_composed_axes(self) -> bool:
        # this will ignore 1 inside brackets
        for axes in self.composition:
            if isinstance(axes, list) and len(axes) > 1:
                return True
        return False

    @staticmethod
    def check_axis_name_return_reason(name: str, allow_underscore: bool = False) -> Tuple[bool, str]:
        if not str.isidentifier(name):
            return False, "not a valid python identifier"
        elif name[0] == "_" or name[-1] == "_":
            if name == "_" and allow_underscore:
                return True, ""
            return False, "axis name should should not start or end with underscore"
        else:
            if keyword.iskeyword(name):
                warnings.warn("It is discouraged to use axes names that are keywords: {}".format(name), RuntimeWarning)
            if name in ["axis"]:
                warnings.warn(
                    "It is discouraged to use 'axis' as an axis name " "and will raise an error in future",
                    FutureWarning,
                )
            return True, ""

    @staticmethod
    def check_axis_name(name: str) -> bool:
        """
        Valid axes names are python identifiers except keywords,
        and additionally should not start or end with underscore
        """
        is_valid, _reason = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid

"""
Backends in `einops` are organized to meet the following requirements
- backends are not imported unless those are actually needed, because
    - backends may not be installed
    - importing all available backends will drive to significant memory footprint
    - backends may be present but installed with errors (but never used),
      importing may drive to crashes
- backend should be either symbolic or imperative
    - this determines which methods (from_numpy/to_numpy or create_symbol/eval_symbol) should be defined
- if backend can't provide symbols for shape dimensions, UnknownSize objects are used
"""

def get_backend(tensor) -> "AbstractBackend":
    """
    Takes a correct backend (e.g. numpy backend if tensor is numpy.ndarray) for a tensor.
    If needed, imports package and creates backend
    """
    _type = type(tensor)
    _result = _type2backend.get(_type, None)
    if _result is not None:
        return _result

    for framework_name, backend in list(_loaded_backends.items()):
        if backend.is_appropriate_type(tensor):
            _type2backend[_type] = backend
            return backend

    # Find backend subclasses recursively
    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)

    for BackendSubclass in backend_subclasses:
        if _debug_importing:
            print("Testing for subclass of ", BackendSubclass)
        if BackendSubclass.framework_name not in _loaded_backends:
            # check that module was already imported. Otherwise it can't be imported
            if BackendSubclass.framework_name in sys.modules:
                if _debug_importing:
                    print("Imported backend for ", BackendSubclass.framework_name)
                backend = BackendSubclass()
                _loaded_backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    _type2backend[_type] = backend
                    return backend

    raise RuntimeError("Tensor type unknown to einops {}".format(type(tensor)))


class AbstractBackend:
    """Base backend class, major part of methods are only for debugging purposes."""

    framework_name: str

    def is_appropriate_type(self, tensor):
        """helper method should recognize tensors it can handle"""
        raise NotImplementedError()

    def from_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def to_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def create_symbol(self, shape):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def eval_symbol(self, symbol, input_dict):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def arange(self, start, stop):
        # supplementary method used only in testing, so should implement CPU version
        raise NotImplementedError("framework doesn't implement arange")

    def shape(self, x):
        """shape should return a tuple with integers or "shape symbols" (which will evaluate to actual size)"""
        return x.shape

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return x.transpose(axes)

    def reduce(self, x, operation, axes):
        return getattr(x, operation)(axis=axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        raise NotImplementedError()

    def add_axis(self, x, new_position):
        raise NotImplementedError()

    def add_axes(self, x, n_axes, pos2len):
        repeats = [1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return self.tile(x, tuple(repeats))

    def tile(self, x, repeats):
        """repeats - same lengths as x.shape"""
        raise NotImplementedError()

    def concat(self, tensors, axis: int):
        """concatenates tensors along axis.
        Assume identical across tensors: devices, dtypes and shapes except selected axis."""
        raise NotImplementedError()

    def is_float_type(self, x):
        # some backends (torch) can't compute average for non-floating types.
        # Decided to drop average for all backends if type is not floating
        raise NotImplementedError()

    def layers(self):
        raise NotImplementedError("backend does not provide layers")

    def __repr__(self):
        return "<einops backend for {}>".format(self.framework_name)

    def einsum(self, pattern, *x):
        raise NotImplementedError("backend does not support einsum")


class UnknownSize:
    """pseudo-symbol for symbolic frameworks which do not provide symbols for shape elements"""

    def __floordiv__(self, other):
        return self

    def __eq__(self, other):
        return True  # we don't know actual size

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __hash__(self):
        return hash(None)


class NumpyBackend(AbstractBackend):
    framework_name = "numpy"

    def __init__(self):
        import numpy

        self.np = numpy

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.np.ndarray)

    def from_numpy(self, x):
        return x

    def to_numpy(self, x):
        return x

    def arange(self, start, stop):
        return self.np.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.np.stack(tensors)

    def tile(self, x, repeats):
        return self.np.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.np.concatenate(tensors, axis=axis)

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

    def add_axis(self, x, new_position):
        return self.np.expand_dims(x, new_position)

    def einsum(self, pattern, *x):
        return self.np.einsum(pattern, *x)


class HashableTuple:
    """Overcomes non-hashability of symbolic elements"""

    def __init__(self, elements: tuple):
        self.elements = elements

    def __iter__(self):
        for x in self.elements:
            yield x

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

    # default equality and hash is used (True only with itself, hash taken of id)


class TensorflowBackend(AbstractBackend):
    framework_name = "tensorflow"

    def __init__(self):
        import tensorflow

        self.tf = tensorflow

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, (self.tf.Tensor, self.tf.Variable))

    def from_numpy(self, x):
        assert self.tf.executing_eagerly()
        return self.tf.convert_to_tensor(x)

    def to_numpy(self, x):
        assert self.tf.executing_eagerly()
        return x.numpy()

    def arange(self, start, stop):
        return self.tf.range(start, stop)

    def shape(self, x):
        if self.tf.executing_eagerly():
            return tuple(UnknownSize() if d is None else int(d) for d in x.shape)
        else:
            static_shape = x.shape.as_list()
            tf_shape = self.tf.shape(x)
            # use the static shape where known, otherwise use the TF shape components
            shape = tuple([s or tf_shape[dim] for dim, s in enumerate(static_shape)])
            try:
                hash(shape)
                return shape
            except BaseException:
                # unhashable symbols in shape. Wrap tuple to be hashable.
                return HashableTuple(shape)

    def reduce(self, x, operation, axes):
        return getattr(self.tf, "reduce_" + operation)(x, axis=axes)

    def reshape(self, x, shape):
        return self.tf.reshape(x, shape)

    def transpose(self, x, axes):
        return self.tf.transpose(x, axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.tf.stack(tensors)

    def tile(self, x, repeats):
        return self.tf.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.tf.concat(tensors, axis=axis)

    def add_axis(self, x, new_position):
        return self.tf.expand_dims(x, new_position)

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

    def layers(self):
        from .layers import tensorflow

        return tensorflow

    def einsum(self, pattern, *x):
        return self.tf.einsum(pattern, *x)


class TFKerasBackend(AbstractBackend):
    framework_name = "tensorflow.keras"

    def __init__(self):
        import tensorflow as tf

        self.tf = tf
        self.keras = tf.keras
        self.K = tf.keras.backend

    def is_appropriate_type(self, tensor):
        return self.tf.is_tensor(tensor) and self.K.is_keras_tensor(tensor)

    def create_symbol(self, shape):
        return self.keras.Input(batch_shape=shape)

    def eval_symbol(self, symbol, input_dict):
        model = self.keras.models.Model([var for (var, _) in input_dict], symbol)
        return model.predict_on_batch([val for (_, val) in input_dict])

    def arange(self, start, stop):
        return self.K.arange(start, stop)

    def shape(self, x):
        shape = self.K.shape(x)  # tf tensor
        return HashableTuple(tuple(shape))

    def reduce(self, x, operation, axes):
        return getattr(self.K, operation)(x, axis=axes)

    def reshape(self, x, shape):
        return self.K.reshape(x, shape)

    def transpose(self, x, axes):
        return self.K.permute_dimensions(x, axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.K.stack(tensors)

    def tile(self, x, repeats):
        return self.K.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.K.concatenate(tensors, axis=axis)

    def add_axis(self, x, new_position):
        return self.K.expand_dims(x, new_position)

    def is_float_type(self, x):
        return "float" in self.K.dtype(x)

    def layers(self):
        from .layers import keras

        return keras


class OneFlowBackend(AbstractBackend):
    framework_name = "oneflow"

    def __init__(self):
        import oneflow as flow

        self.flow = flow

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.flow.Tensor)

    def from_numpy(self, x):
        variable = self.flow.from_numpy(x)
        if self.is_float_type(variable):
            # attach grad only to floating types
            variable.requires_grad = True
        return variable

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def arange(self, start, stop):
        return self.flow.arange(start, stop, dtype=self.flow.int64)

    def reduce(self, x, operation, reduced_axes):
        for axis in sorted(reduced_axes, reverse=True):
            if operation == "min":
                x, _ = x.min(dim=axis)
            elif operation == "max":
                x, _ = x.max(dim=axis)
            elif operation in ["sum", "mean", "prod", "any", "all"]:
                x = getattr(x, operation)(dim=axis)
            else:
                raise NotImplementedError("Unknown reduction ", operation)
        return x

    def transpose(self, x, axes):
        return x.permute(axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.flow.stack(tensors)

    def add_axes(self, x, n_axes, pos2len):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(*repeats)

    def tile(self, x, repeats):
        return x.repeat(repeats)

    def concat(self, tensors, axis: int):
        return self.flow.concat(tensors, dim=axis)

    def add_axis(self, x, new_position):
        return self.flow.unsqueeze(x, new_position)

    def is_float_type(self, x):
        return x.dtype in [self.flow.float16, self.flow.float32, self.flow.float64]

    def layers(self):
        from .layers import oneflow

        return oneflow

    def einsum(self, pattern, *x):
        return self.flow.einsum(pattern, *x)


class PaddleBackend(AbstractBackend):
    framework_name = "paddle"

    def __init__(self):
        import paddle

        self.paddle = paddle

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, (self.paddle.Tensor, self.paddle.static.Variable))

    def from_numpy(self, x):
        tensor = self.paddle.to_tensor(x)
        tensor.stop_gradient = False
        return tensor

    def to_numpy(self, x):
        return x.detach().numpy()

    def arange(self, start, stop):
        return self.paddle.arange(start, stop, dtype=self.paddle.int64)

    def reduce(self, x, operation, axes):
        if len(axes) == x.ndim:
            # currently paddle returns 1d tensor instead of 0d
            return super().reduce(x, operation, axes).squeeze(0)
        else:
            return super().reduce(x, operation, axes)

    def transpose(self, x, axes):
        return x.transpose(axes)

    def add_axes(self, x, n_axes, pos2len):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.paddle.stack(tensors)

    def reshape(self, x, shape):
        return x.reshape(shape)

    def tile(self, x, repeats):
        return x.tile(repeats)

    def concat(self, tensors, axis: int):
        return self.paddle.concat(tensors, axis=axis)

    def add_axis(self, x, new_position):
        return x.unsqueeze(new_position)

    def is_float_type(self, x):
        return x.dtype in [self.paddle.float16, self.paddle.float32, self.paddle.float64]

    def layers(self):
        from .layers import paddle

        return paddle

    def einsum(self, pattern, *x):
        return self.paddle.einsum(pattern, *x)

    def shape(self, x):
        return tuple(x.shape)


class TinygradBackend(AbstractBackend):
    framework_name = "tinygrad"

    def __init__(self):
        import tinygrad

        self.tinygrad = tinygrad

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.tinygrad.Tensor)

    def from_numpy(self, x):
        return self.tinygrad.Tensor(x)

    def to_numpy(self, x):
        return x.numpy()

    def arange(self, start, stop):
        return self.tinygrad.Tensor.arange(start, stop)

    def shape(self, x):
        return x.shape

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return x.permute(axes)

    def reduce(self, x, operation, axes):
        for axis in sorted(axes, reverse=True):
            x = getattr(x, operation)(axis=axis)
        return x

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.tinygrad.Tensor.stack(tensors)

    def add_axis(self, x, new_position):
        return x.unsqueeze(new_position)

    def tile(self, x, repeats):
        return x.repeat(repeats)

    def concat(self, tensors, axis: int):
        return tensors[0].cat(tensors[1:], axis) if len(tensors) > 1 else tensors[0]

    def is_float_type(self, x):
        return self.tinygrad.dtypes.is_float(x.dtype)

    def einsum(self, pattern, *x):
        return self.tinygrad.Tensor.einsum(pattern, *x)



if typing.TYPE_CHECKING:
    # for docstrings in pycharm
    import numpy as np  # noqa E401

Tensor = TypeVar("Tensor")
ReductionCallable = Callable[[Tensor, Tuple[int, ...]], Tensor]
Reduction = Union[str, ReductionCallable]

_reductions = ("min", "max", "sum", "mean", "prod", "any", "all")

# magic integers are required to stay within
# traceable subset of language
_unknown_axis_length = -999999
_expected_axis_length = -99999


def _product(sequence: List[int]) -> int:
    """minimalistic product that works both with numbers and symbols. Supports empty lists"""
    result = 1
    for element in sequence:
        result *= element
    return result


def _reduce_axes(tensor, reduction_type: Reduction, reduced_axes: List[int], backend):
    if callable(reduction_type):
        # custom callable
        return reduction_type(tensor, tuple(reduced_axes))
    else:
        # one of built-in operations
        assert reduction_type in _reductions
        if reduction_type == "mean":
            if not backend.is_float_type(tensor):
                raise NotImplementedError("reduce_mean is not available for non-floating tensors")
        return backend.reduce(tensor, reduction_type, tuple(reduced_axes))


def _optimize_transformation(init_shapes, reduced_axes, axes_reordering, final_shapes):
    # 'collapses' neighboring axes if those participate in the result pattern in the same order
    # TODO add support for added_axes
    assert len(axes_reordering) + len(reduced_axes) == len(init_shapes)
    # joining consecutive axes that will be reduced
    # possibly we can skip this if all backends can optimize this (not sure)
    reduced_axes = tuple(sorted(reduced_axes))
    for i in range(len(reduced_axes) - 1)[::-1]:
        if reduced_axes[i] + 1 == reduced_axes[i + 1]:
            removed_axis = reduced_axes[i + 1]
            removed_length = init_shapes[removed_axis]
            init_shapes = init_shapes[:removed_axis] + init_shapes[removed_axis + 1 :]
            init_shapes[removed_axis - 1] *= removed_length
            reduced_axes = reduced_axes[: i + 1] + tuple(axis - 1 for axis in reduced_axes[i + 2 :])

    # removing axes that are moved together during reshape
    def build_mapping():
        init_to_final = {}
        for axis in range(len(init_shapes)):
            if axis in reduced_axes:
                init_to_final[axis] = None
            else:
                after_reduction = sum(x is not None for x in init_to_final.values())
                init_to_final[axis] = list(axes_reordering).index(after_reduction)
        return init_to_final

    init_axis_to_final_axis = build_mapping()

    for init_axis in range(len(init_shapes) - 1)[::-1]:
        if init_axis_to_final_axis[init_axis] is None:
            continue
        if init_axis_to_final_axis[init_axis + 1] is None:
            continue
        if init_axis_to_final_axis[init_axis] + 1 == init_axis_to_final_axis[init_axis + 1]:
            removed_axis = init_axis + 1
            removed_length = init_shapes[removed_axis]
            removed_axis_after_reduction = sum(x not in reduced_axes for x in range(removed_axis))

            reduced_axes = tuple(axis if axis < removed_axis else axis - 1 for axis in reduced_axes)
            init_shapes = init_shapes[:removed_axis] + init_shapes[removed_axis + 1 :]
            init_shapes[removed_axis - 1] *= removed_length
            old_reordering = axes_reordering
            axes_reordering = []
            for axis in old_reordering:
                if axis == removed_axis_after_reduction:
                    pass
                elif axis < removed_axis_after_reduction:
                    axes_reordering.append(axis)
                else:
                    axes_reordering.append(axis - 1)
            init_axis_to_final_axis = build_mapping()

    return init_shapes, reduced_axes, axes_reordering, final_shapes


CookedRecipe = Tuple[Optional[List[int]], Optional[List[int]], List[int], Dict[int, int], Optional[List[int]], int]

# Actual type is tuple[tuple[str, int], ...]
# However torch.jit.script does not "understand" the correct type,
# and torch_specific will use list version.
HashableAxesLengths = Tuple[Tuple[str, int], ...]
FakeHashableAxesLengths = List[Tuple[str, int]]


class TransformRecipe:
    """
    Recipe describes actual computation pathway.
    Recipe can be applied to a tensor or variable.
    """

    # structure is non-mutable. In future, this can be non-mutable dataclass (python 3.7+)
    # update: pytorch 2.0 torch.jit.script seems to have problems with dataclasses unless they were explicitly provided

    def __init__(
        self,
        # list of sizes (or just sizes) for elementary axes as they appear in left expression.
        # this is what (after computing unknown parts) will be a shape after first transposition.
        # This does not include any ellipsis dimensions.
        elementary_axes_lengths: List[int],
        # if additional axes are provided, they should be set in prev array
        # This shows mapping from name to position
        axis_name2elementary_axis: Dict[str, int],
        # each dimension in input can help to reconstruct length of one elementary axis
        # or verify one of dimensions. Each element points to element of elementary_axes_lengths.
        input_composition_known_unknown: List[Tuple[List[int], List[int]]],
        # permutation applied to elementary axes, if ellipsis is absent
        axes_permutation: List[int],
        # permutation puts reduced axes in the end, we only need to know the first position.
        first_reduced_axis: int,
        # at which positions which of elementary axes should appear. Axis position -> axis index.
        added_axes: Dict[int, int],
        # ids of axes as they appear in result, again pointers to elementary_axes_lengths,
        # only used to infer result dimensions
        output_composite_axes: List[List[int]],
    ):
        self.elementary_axes_lengths: List[int] = elementary_axes_lengths
        self.axis_name2elementary_axis: Dict[str, int] = axis_name2elementary_axis
        self.input_composition_known_unknown: List[Tuple[List[int], List[int]]] = input_composition_known_unknown
        self.axes_permutation: List[int] = axes_permutation

        self.first_reduced_axis: int = first_reduced_axis
        self.added_axes: Dict[int, int] = added_axes
        self.output_composite_axes: List[List[int]] = output_composite_axes


def _reconstruct_from_shape_uncached(
    self: TransformRecipe, shape: List[int], axes_dims: FakeHashableAxesLengths
) -> CookedRecipe:
    """
    Reconstruct all actual parameters using shape.
    Shape is a tuple that may contain integers, shape symbols (tf, theano) and UnknownSize (tf, previously mxnet)
    known axes can be integers or symbols, but not Nones.
    """
    # magic number
    need_init_reshape = False

    # last axis is allocated for collapsed ellipsis
    axes_lengths: List[int] = list(self.elementary_axes_lengths)
    for axis, dim in axes_dims:
        axes_lengths[self.axis_name2elementary_axis[axis]] = dim

    for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composition_known_unknown):
        length = shape[input_axis]
        if len(known_axes) == 0 and len(unknown_axes) == 1:
            # shortcut for the most common case
            axes_lengths[unknown_axes[0]] = length
            continue

        known_product = 1
        for axis in known_axes:
            known_product *= axes_lengths[axis]

        if len(unknown_axes) == 0:
            if isinstance(length, int) and isinstance(known_product, int) and length != known_product:
                raise ValueError(f"Shape mismatch, {length} != {known_product}")
        else:
            # assert len(unknown_axes) == 1, 'this is enforced when recipe is created, so commented out'
            if isinstance(length, int) and isinstance(known_product, int) and length % known_product != 0:
                raise ValueError(f"Shape mismatch, can't divide axis of length {length} in chunks of {known_product}")

            unknown_axis = unknown_axes[0]
            inferred_length: int = length // known_product
            axes_lengths[unknown_axis] = inferred_length

        if len(known_axes) + len(unknown_axes) != 1:
            need_init_reshape = True

    # at this point all axes_lengths are computed (either have values or variables, but not Nones)

    # elementary axes are ordered as they appear in input, then all added axes
    init_shapes: Optional[List[int]] = axes_lengths[: len(self.axes_permutation)] if need_init_reshape else None

    need_final_reshape = False
    final_shapes: List[int] = []
    for grouping in self.output_composite_axes:
        lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
        final_shapes.append(_product(lengths))
        if len(lengths) != 1:
            need_final_reshape = True

    added_axes: Dict[int, int] = {
        pos: axes_lengths[pos_in_elementary] for pos, pos_in_elementary in self.added_axes.items()
    }

    # this list can be empty
    reduced_axes = list(range(self.first_reduced_axis, len(self.axes_permutation)))

    n_axes_after_adding_axes = len(added_axes) + len(self.axes_permutation)

    axes_reordering: Optional[List[int]] = self.axes_permutation
    if self.axes_permutation == list(range(len(self.axes_permutation))):
        axes_reordering = None

    _final_shapes = final_shapes if need_final_reshape else None
    return init_shapes, axes_reordering, reduced_axes, added_axes, _final_shapes, n_axes_after_adding_axes


_reconstruct_from_shape = functools.lru_cache(1024)(_reconstruct_from_shape_uncached)


def _apply_recipe(
    backend, recipe: TransformRecipe, tensor: Tensor, reduction_type: Reduction, axes_lengths: HashableAxesLengths
) -> Tensor:
    # this method implements actual work for all backends for 3 operations
    try:
        init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added = _reconstruct_from_shape(
            recipe, backend.shape(tensor), axes_lengths
        )
    except TypeError:
        # shape or one of passed axes lengths is not hashable (i.e. they are symbols)
        _result = _reconstruct_from_shape_uncached(recipe, backend.shape(tensor), axes_lengths)
        (init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added) = _result
    if init_shapes is not None:
        tensor = backend.reshape(tensor, init_shapes)
    if axes_reordering is not None:
        tensor = backend.transpose(tensor, axes_reordering)
    if len(reduced_axes) > 0:
        tensor = _reduce_axes(tensor, reduction_type=reduction_type, reduced_axes=reduced_axes, backend=backend)
    if len(added_axes) > 0:
        tensor = backend.add_axes(tensor, n_axes=n_axes_w_added, pos2len=added_axes)
    if final_shapes is not None:
        tensor = backend.reshape(tensor, final_shapes)
    return tensor


def _apply_recipe_array_api(
    xp, recipe: TransformRecipe, tensor: Tensor, reduction_type: Reduction, axes_lengths: HashableAxesLengths
) -> Tensor:
    # completely-inline implementation
    init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added = _reconstruct_from_shape(
        recipe, tensor.shape, axes_lengths
    )
    if init_shapes is not None:
        tensor = xp.reshape(tensor, init_shapes)
    if axes_reordering is not None:
        tensor = xp.permute_dims(tensor, axes_reordering)
    if len(reduced_axes) > 0:
        if callable(reduction_type):
            # custom callable
            tensor = reduction_type(tensor, tuple(reduced_axes))
        else:
            # one of built-in operations
            assert reduction_type in _reductions
            tensor = getattr(xp, reduction_type)(tensor, axis=tuple(reduced_axes))
    if len(added_axes) > 0:
        # we use broadcasting
        for axis_position, axis_length in added_axes.items():
            tensor = xp.expand_dims(tensor, axis=axis_position)

        final_shape = list(tensor.shape)
        for axis_position, axis_length in added_axes.items():
            final_shape[axis_position] = axis_length

        tensor = xp.broadcast_to(tensor, final_shape)
    if final_shapes is not None:
        tensor = xp.reshape(tensor, final_shapes)
    return tensor


@functools.lru_cache(256)
def _prepare_transformation_recipe(
    pattern: str,
    operation: Reduction,
    axes_names: Tuple[str, ...],
    ndim: int,
) -> TransformRecipe:
    """Perform initial parsing of pattern and provided supplementary info
    axes_lengths is a tuple of tuples (axis_name, axis_length)
    """
    left_str, rght_str = pattern.split("->")
    left = ParsedExpression(left_str)
    rght = ParsedExpression(rght_str)

    # checking that axes are in agreement - new axes appear only in repeat, while disappear only in reduction
    if not left.has_ellipsis and rght.has_ellipsis:
        raise ValueError("Ellipsis found in right side, but not left side of a pattern {}".format(pattern))
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise ValueError("Ellipsis inside parenthesis in the left side is not allowed: {}".format(pattern))
    if operation == "rearrange":
        if left.has_non_unitary_anonymous_axes or rght.has_non_unitary_anonymous_axes:
            raise ValueError("Non-unitary anonymous axes are not supported in rearrange (exception is length 1)")
        difference = set.symmetric_difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise ValueError("Identifiers only on one side of expression (should be on both): {}".format(difference))
    elif operation == "repeat":
        difference = set.difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise ValueError("Unexpected identifiers on the left side of repeat: {}".format(difference))
        axes_without_size = set.difference(
            {ax for ax in rght.identifiers if not isinstance(ax, AnonymousAxis)},
            {*left.identifiers, *axes_names},
        )
        if len(axes_without_size) > 0:
            raise ValueError("Specify sizes for new axes in repeat: {}".format(axes_without_size))
    elif operation in _reductions or callable(operation):
        difference = set.difference(rght.identifiers, left.identifiers)
        if len(difference) > 0:
            raise ValueError("Unexpected identifiers on the right side of reduce {}: {}".format(operation, difference))
    else:
        raise ValueError("Unknown reduction {}. Expect one of {}.".format(operation, _reductions))

    if left.has_ellipsis:
        n_other_dims = len(left.composition) - 1
        if ndim < n_other_dims:
            raise ValueError(f"Wrong shape: expected >={n_other_dims} dims. Received {ndim}-dim tensor.")
        ellipsis_ndim = ndim - n_other_dims
        ell_axes = [_ellipsis + str(i) for i in range(ellipsis_ndim)]
        left_composition = []
        for composite_axis in left.composition:
            if composite_axis == _ellipsis:
                for axis in ell_axes:
                    left_composition.append([axis])
            else:
                left_composition.append(composite_axis)

        rght_composition = []
        for composite_axis in rght.composition:
            if composite_axis == _ellipsis:
                for axis in ell_axes:
                    rght_composition.append([axis])
            else:
                group = []
                for axis in composite_axis:
                    if axis == _ellipsis:
                        group.extend(ell_axes)
                    else:
                        group.append(axis)
                rght_composition.append(group)

        left.identifiers.update(ell_axes)
        left.identifiers.remove(_ellipsis)
        if rght.has_ellipsis:
            rght.identifiers.update(ell_axes)
            rght.identifiers.remove(_ellipsis)
    else:
        if ndim != len(left.composition):
            raise ValueError(f"Wrong shape: expected {len(left.composition)} dims. Received {ndim}-dim tensor.")
        left_composition = left.composition
        rght_composition = rght.composition

    # parsing all dimensions to find out lengths
    axis_name2known_length: Dict[Union[str, AnonymousAxis], int] = OrderedDict()
    for composite_axis in left_composition:
        for axis_name in composite_axis:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length

    # axis_ids_after_first_reshape = range(len(axis_name2known_length)) at this point

    repeat_axes_names = []
    for axis_name in rght.identifiers:
        if axis_name not in axis_name2known_length:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
            repeat_axes_names.append(axis_name)

    axis_name2position = {name: position for position, name in enumerate(axis_name2known_length)}

    # axes provided as kwargs
    for elementary_axis in axes_names:
        if not ParsedExpression.check_axis_name(elementary_axis):
            raise ValueError("Invalid name for an axis", elementary_axis)
        if elementary_axis not in axis_name2known_length:
            raise ValueError("Axis {} is not used in transform".format(elementary_axis))
        axis_name2known_length[elementary_axis] = _expected_axis_length

    input_axes_known_unknown = []
    # some shapes are inferred later - all information is prepared for faster inference
    for i, composite_axis in enumerate(left_composition):
        known: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] != _unknown_axis_length}
        unknown: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] == _unknown_axis_length}
        if len(unknown) > 1:
            raise ValueError("Could not infer sizes for {}".format(unknown))
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(
            ([axis_name2position[axis] for axis in known], [axis_name2position[axis] for axis in unknown])
        )

    axis_position_after_reduction: Dict[str, int] = {}
    for axis_name in itertools.chain(*left_composition):
        if axis_name in rght.identifiers:
            axis_position_after_reduction[axis_name] = len(axis_position_after_reduction)

    result_axes_grouping: List[List[int]] = [
        [axis_name2position[axis] for axis in composite_axis] for i, composite_axis in enumerate(rght_composition)
    ]

    ordered_axis_left = list(itertools.chain(*left_composition))
    ordered_axis_rght = list(itertools.chain(*rght_composition))
    reduced_axes = [axis for axis in ordered_axis_left if axis not in rght.identifiers]
    order_after_transposition = [axis for axis in ordered_axis_rght if axis in left.identifiers] + reduced_axes
    axes_permutation = [ordered_axis_left.index(axis) for axis in order_after_transposition]
    added_axes = {
        i: axis_name2position[axis_name]
        for i, axis_name in enumerate(ordered_axis_rght)
        if axis_name not in left.identifiers
    }

    first_reduced_axis = len(order_after_transposition) - len(reduced_axes)

    return TransformRecipe(
        elementary_axes_lengths=list(axis_name2known_length.values()),
        axis_name2elementary_axis={axis: axis_name2position[axis] for axis in axes_names},
        input_composition_known_unknown=input_axes_known_unknown,
        axes_permutation=axes_permutation,
        first_reduced_axis=first_reduced_axis,
        added_axes=added_axes,
        output_composite_axes=result_axes_grouping,
    )


def _prepare_recipes_for_all_dims(
    pattern: str, operation: Reduction, axes_names: Tuple[str, ...]
) -> Dict[int, TransformRecipe]:
    """
    Internal function, used in layers.
    Layer makes all recipe creation when it is initialized, thus to keep recipes simple we pre-compute for all dims
    """
    left_str, rght_str = pattern.split("->")
    left = ParsedExpression(left_str)
    dims = [len(left.composition)]
    if left.has_ellipsis:
        dims = [len(left.composition) - 1 + ellipsis_dims for ellipsis_dims in range(8)]
    return {ndim: _prepare_transformation_recipe(pattern, operation, axes_names, ndim=ndim) for ndim in dims}


def reduce(tensor: Union[Tensor, List[Tensor]], pattern: str, reduction: Reduction, **axes_lengths: int) -> Tensor:
    """
    einops.reduce provides combination of reordering and reduction using reader-friendly notation.

    Examples for reduce operation:

    ```python
    >>> x = np.random.randn(100, 32, 64)

    # perform max-reduction on the first axis
    >>> y = reduce(x, 't b c -> b c', 'max')

    # same as previous, but with clearer axes meaning
    >>> y = reduce(x, 'time batch channel -> batch channel', 'max')

    >>> x = np.random.randn(10, 20, 30, 40)

    # 2d max-pooling with kernel size = 2 * 2 for image processing
    >>> y1 = reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2)

    # if one wants to go back to the original height and width, depth-to-space trick can be applied
    >>> y2 = rearrange(y1, 'b (c h2 w2) h1 w1 -> b c (h1 h2) (w1 w2)', h2=2, w2=2)
    >>> assert parse_shape(x, 'b _ h w') == parse_shape(y2, 'b _ h w')

    # Adaptive 2d max-pooling to 3 * 4 grid
    >>> reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h1=3, w1=4).shape
    (10, 20, 3, 4)

    # Global average pooling
    >>> reduce(x, 'b c h w -> b c', 'mean').shape
    (10, 20)

    # Subtracting mean over batch for each channel
    >>> y = x - reduce(x, 'b c h w -> () c () ()', 'mean')

    # Subtracting per-image mean for each channel
    >>> y = x - reduce(x, 'b c h w -> b c () ()', 'mean')

    ```

    Parameters:
        tensor: tensor: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
            list of tensors is also accepted, those should be of the same type and shape
        pattern: string, reduction pattern
        reduction: one of available reductions ('min', 'max', 'sum', 'mean', 'prod'), case-sensitive
            alternatively, a callable f(tensor, reduced_axes) -> tensor can be provided.
            This allows using various reductions, examples: np.max, tf.reduce_logsumexp, torch.var, etc.
        axes_lengths: any additional specifications for dimensions

    Returns:
        tensor of the same type as input
    """
    try:
        if isinstance(tensor, list):
            if len(tensor) == 0:
                raise TypeError("Rearrange/Reduce/Repeat can't be applied to an empty list")
            backend = get_backend(tensor[0])
            tensor = backend.stack_on_zeroth_dimension(tensor)
        else:
            backend = get_backend(tensor)

        hashable_axes_lengths = tuple(axes_lengths.items())
        shape = backend.shape(tensor)
        recipe = _prepare_transformation_recipe(pattern, reduction, axes_names=tuple(axes_lengths), ndim=len(shape))
        return _apply_recipe(
            backend, recipe, cast(Tensor, tensor), reduction_type=reduction, axes_lengths=hashable_axes_lengths
        )
    except ValueError as e:
        message = ' Error while processing {}-reduction pattern "{}".'.format(reduction, pattern)
        if not isinstance(tensor, list):
            message += "\n Input tensor shape: {}. ".format(shape)
        else:
            message += "\n Input is list. "
        message += "Additional info: {}.".format(axes_lengths)
        raise ValueError(message + "\n {}".format(e))


def rearrange(tensor: Union[Tensor, List[Tensor]], pattern: str, **axes_lengths) -> Tensor:
    """
    einops.rearrange is a reader-friendly smart element reordering for multidimensional tensors.
    This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    Examples for rearrange operation:

    ```python
    # suppose we have a set of 32 images in "h w c" format (height-width-channel)
    >>> images = [np.random.randn(30, 40, 3) for _ in range(32)]

    # stack along first (batch) axis, output is a single array
    >>> rearrange(images, 'b h w c -> b h w c').shape
    (32, 30, 40, 3)

    # concatenate images along height (vertical axis), 960 = 32 * 30
    >>> rearrange(images, 'b h w c -> (b h) w c').shape
    (960, 40, 3)

    # concatenated images along horizontal axis, 1280 = 32 * 40
    >>> rearrange(images, 'b h w c -> h (b w) c').shape
    (30, 1280, 3)

    # reordered axes to "b c h w" format for deep learning
    >>> rearrange(images, 'b h w c -> b c h w').shape
    (32, 3, 30, 40)

    # flattened each image into a vector, 3600 = 30 * 40 * 3
    >>> rearrange(images, 'b h w c -> b (c h w)').shape
    (32, 3600)

    # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
    >>> rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape
    (128, 15, 20, 3)

    # space-to-depth operation
    >>> rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape
    (32, 15, 20, 12)

    ```

    When composing axes, C-order enumeration used (consecutive elements have different last axis)
    Find more examples in einops tutorial.

    Parameters:
        tensor: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
                list of tensors is also accepted, those should be of the same type and shape
        pattern: string, rearrangement pattern
        axes_lengths: any additional specifications for dimensions

    Returns:
        tensor of the same type as input. If possible, a view to the original tensor is returned.

    """
    return reduce(tensor, pattern, reduction="rearrange", **axes_lengths)


def repeat(tensor: Union[Tensor, List[Tensor]], pattern: str, **axes_lengths) -> Tensor:
    """
    einops.repeat allows reordering elements and repeating them in arbitrary combinations.
    This operation includes functionality of repeat, tile, broadcast functions.

    Examples for repeat operation:

    ```python
    # a grayscale image (of shape height x width)
    >>> image = np.random.randn(30, 40)

    # change it to RGB format by repeating in each channel
    >>> repeat(image, 'h w -> h w c', c=3).shape
    (30, 40, 3)

    # repeat image 2 times along height (vertical axis)
    >>> repeat(image, 'h w -> (repeat h) w', repeat=2).shape
    (60, 40)

    # repeat image 2 time along height and 3 times along width
    >>> repeat(image, 'h w -> (h2 h) (w3 w)', h2=2, w3=3).shape
    (60, 120)

    # convert each pixel to a small square 2x2. Upsample image by 2x
    >>> repeat(image, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
    (60, 80)

    # pixelate image first by downsampling by 2x, then upsampling
    >>> downsampled = reduce(image, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
    >>> repeat(downsampled, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
    (30, 40)

    ```

    When composing axes, C-order enumeration used (consecutive elements have different last axis)
    Find more examples in einops tutorial.

    Parameters:
        tensor: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
            list of tensors is also accepted, those should be of the same type and shape
        pattern: string, rearrangement pattern
        axes_lengths: any additional specifications for dimensions

    Returns:
        Tensor of the same type as input. If possible, a view to the original tensor is returned.

    """
    return reduce(tensor, pattern, reduction="repeat", **axes_lengths)


