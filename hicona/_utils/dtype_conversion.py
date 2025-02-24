"""Dictionaries for dtype conversion across different formats.

In all mappings:
- The key is the source dtype.
- The value is the target dtype.
- If the key is commented out, it means that the target format does not support that dtype.
- If there is a comment after the values, it means that no 1-to-1 mapping is possible, but
  since the conversion is needed, the comment explains how it is done (usually upcasting).
"""

# Mapping from numpy to graph-tool types
# Reverse should not be needed, since graph-tool can yield numpy arrays.

# Only numeric types from the following links are considered
# https://numpy.org/doc/stable/user/basics.types.html
# https://graph-tool.skewed.de/static/doc/quickstart.html#property-maps

# NOTE: vector types might be added in the future
# NOTE: np has fixed sized strings, but converting polars of pandas str columns to numpy
#       will result in object dtype, rather than "<UX" etc. Using object dtype for now.

NP_TO_GT: dict[str, str] = {
    "object": "string",  # See comment above
    "bool": "uint8_t",  # Upcast to 8-bit unsigned integer
    "int8": "int16_t",  # Upcast to 16-bit integer
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "float16": "double",  # Upcast to double
    "float32": "double",  # Upcast to double
    "float64": "double",  # Upcast to double
    "float128": "long double",
    # "uint8": "",
    # "uint16": "",
    "uint32": "int32_t",
    # "uint64": "",
    # "complex64": "",
    # "complex128": "",
    # "complex256": "",
    # "intp": "",
    # "uintp": "",
}

GT_TO_NP: dict[str, str] = {
    "string": "object",
    "uint8_t": "bool",
    "bool": "bool",  # bool is an alaias of uint8_t in graph-tool
    "int16_t": "int16",
    "int32_t": "int32",
    "int64_t": "int64",
    "double": "float64",
}
