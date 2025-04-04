import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple, List, Optional, Union


# Custom Exception Class for Errors
class Error(Exception):
  pass


# Parse the patterns into input and output axes specifications
def parser(pattern: str) -> Tuple[List[str], List[str]]:
  if '->' not in pattern:
    raise Error("Pattern must contain '->' separator")

  def split(s: str) -> List[str]:
    # Custom split function to get the parenthesis
    result = []
    current = ''
    paren_count = 0

    for char in s.strip():
      if char =='(':
        paren_count += 1
        current += char
      elif char == ')':
        paren_count -= 1
        current += char
      elif char.isspace() and paren_count == 0:
        if current:
          result.append(current)
          current = ''
      else:
        current += char

    if current:
      result.append(current)
    return result

  # Split pattern into input and output specifications
  input, output = pattern.split('->')
  return split(input), split(output)


# Parse a single axis specification into its name and components
def parse_axis(axis: str) -> Tuple[str, Optional[List[str]]]:
  if axis == '...':
    return axis, None
  if '(' in axis and ')' in axis:
    # Handling of composite axes like '(h w)'
    parts = axis.strip('()').split()
    return ''.join(parts), parts
  return axis, None


# Builds a shape dictionary and validates tensor compatibility with the pattern
def validate_shapes(tensor: np.ndarray,
                    input_axes: List[str],
                    output_axes: List[str],
                    axes_length: Dict[str, int]) -> Dict[str, int]:
  shape = list(tensor.shape)
  shape_dict = OrderedDict()
  tensor_rank = len(shape)
  base_input_axes = [ax for ax in input_axes if ax != '...']

  shape_idx = 0
  for axis in input_axes:
    if axis == '...':
      # Handles ellipsis by calculating remaining dimensions
      n_ellipsis = tensor_rank - len(base_input_axes)
      for i in range(n_ellipsis):
        shape_dict[f'_ellipsis_{i}'] = shape[shape_idx]
        shape_idx += 1
      continue
    name, components = parse_axis(axis)
    current_size = shape[shape_idx]

    if components:
      # Validate and process split operations
      known_sizes = {c: axes_length[c] for c in components if c in axes_length}
      if not known_sizes:
        raise Error(f"Must specify at least one dimension for split {axis}")
      known_product = np.prod(list(known_sizes.values()))
      if current_size % known_product != 0:
        raise Error(f"Cannot split {current_size} with sizes {known_sizes}")
      remaining_size = current_size // known_product
      remaining_comps = [c for c in components if c not in known_sizes]
      if len(remaining_comps) > 1:
        raise Error("Can only infer one unknown dimension per split")
      for comp in components:
        shape_dict[comp] = known_sizes.get(comp, remaining_size)
    else:
      shape_dict[name] = current_size
    shape_idx += 1

  # Add any additional axis lengths from kwargs
  for name, size in axes_length.items():
    if name not in shape_dict:
      shape_dict[name] = size

  return shape_dict


# Rearrange a tensor based on operations like reshaping, transposing, splitting, merging, repeating of axes
# Has separate `has_split`, `has_merge` and `ellipsis_present` variables to handle each of the cases properly
def rearrange(tensor: np.ndarray, pattern: str, **axes_lengths) -> np.ndarray:
  # Parse the pattern into input & output axes
  input_axes, output_axes = parser(pattern)

  # Detect operation types
  has_split = any('(' in ax and ')' in ax for ax in input_axes)
  has_merge = any('(' in ax and ')' in ax for ax in output_axes)
  ellipsis_present = '...' in input_axes or '...' in output_axes
  base_input_axes = [ax for ax in input_axes if ax != '...']

  tensor_rank = len(tensor.shape)
  if not has_split and not has_merge:
    # Validate simple transposition
    if ellipsis_present:
      n_ellipsis = tensor_rank - len(base_input_axes)
      if n_ellipsis < 0:
        raise Error("Too many explicit axes for tensor rank")
    else:
      if len(base_input_axes) != tensor_rank:
        raise Error(f"Transpose: Number of axes ({len(base_input_axes)}) doesn't match tensor rank ({tensor_rank})")
  elif has_split:
    if len(base_input_axes) != tensor_rank:
      raise Error(f"Split: Number of base axes ({len(base_input_axes)}) doesn't match tensor rank ({tensor_rank})")

  # Build and validate shape dictionary
  shape_dict = validate_shapes(tensor, input_axes, output_axes, axes_lengths)

  if has_merge:
    # Validate merge operation components
    output_comps = set()
    for ax in output_axes:
      _, comps = parse_axis(ax)
      if comps:
        output_comps.update(comps)
    input_comps = set(shape_dict.keys()) - {k for k in shape_dict if k.startswith('_ellipsis_')}
    if not output_comps.issubset(input_comps):
      raise Error(f"Merge contains unknown components: {output_comps - input_comps}")

  # Validate output axes
  for ax in output_axes:
    name, comps = parse_axis(ax)
    if name != '...' and not comps and name not in shape_dict:
      raise Error(f"Unknown axis {name} in output")

  current_shape = list(tensor.shape)
  ops = []
  axis_map = {}

  # Build operation sequence and axis mapping
  shape_idx = 0
  for i, axis in enumerate(input_axes):
    if axis == '...':
      n_ellipsis = tensor_rank - len(base_input_axes)
      for j in range(n_ellipsis):
        axis_map[f'_ellipsis_{j}'] = shape_idx + j
      shape_idx += n_ellipsis
      continue
    name, components = parse_axis(axis)
    if components:
      split_shape = [shape_dict[comp] for comp in components]
      ops.append(('reshape', (shape_idx, split_shape)))
      for j, comp in enumerate(components):
        axis_map[comp] = shape_idx + j
      shape_idx += len(components)
    else:
      if name == '1' and current_shape[shape_idx] == 1:
        axis_map[name] = shape_idx
      else:
        axis_map[name] = shape_idx
      shape_idx += 1

  # Construct final shape and permutation
  final_shape = []
  permutation = []
  current_axis_pos = 0

  for axis in output_axes:
        name, components = parse_axis(axis)
        if name == '...':
            n_ellipsis = tensor_rank - len(base_input_axes)
            for i in range(n_ellipsis):
                final_shape.append(shape_dict[f'_ellipsis_{i}'])
                permutation.append(axis_map[f'_ellipsis_{i}'])
            current_axis_pos += n_ellipsis
        elif components:
            sizes = [shape_dict[comp] for comp in components]
            final_shape.append(np.prod(sizes))
            perm = [axis_map[comp] for comp in components]
            permutation.extend(perm)
            current_axis_pos += 1
        else:
            if name in axes_lengths and name not in axis_map:
                final_shape.append(axes_lengths[name])
                singleton_pos = [i for i, ax in enumerate(input_axes) if ax == '1' and current_shape[i] == 1][0]
                permutation.append(singleton_pos)
            else:
                final_shape.append(shape_dict[name])
                permutation.append(axis_map[name])
            current_axis_pos += 1

  result = tensor
  # Apply reshape operations for splits
  for op_type, (axis, shape) in ops:
    if op_type == 'reshape':
      new_shape = current_shape[:axis] + shape + current_shape[axis+1:]
      result = result.reshape(new_shape)
      current_shape = new_shape

  if permutation:
    result = np.transpose(result, permutation)

  if final_shape:
    if any(s == 1 for s in current_shape) and not has_split and not has_merge:
      # Handle broadcasting for simple transformations
      broadcast_shape = []
      for ax in output_axes:
        if ax == '...':
          n_ellipsis = tensor_rank - len(base_input_axes)
          broadcast_shape.extend([shape_dict[f'_ellipsis_{i}'] for i in range(n_ellipsis)])
        elif parse_axis(ax)[1]:
          sizes = [shape_dict[c] for c in parse_axis(ax)[1]]
          broadcast_shape.append(np.prod(sizes))
        else:
          broadcast_shape.append(shape_dict.get(ax, axes_lengths.get(ax, 1)))
      result = np.broadcast_to(result, broadcast_shape)
    else:
      # Final reshape to target shape
      result = result.reshape(final_shape)

  return result