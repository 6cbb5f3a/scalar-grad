# C++23 Automatic Differentiation Tensor

A lightweight scalar automatic differentiation project implemented in modern C++23.

The project builds a computational graph to track Tensor operations and uses reverse-mode automatic differentiation to calculate gradients.

## Features

- Shared pointer-based Tensor node management
- Computational graph construction
- Reverse-mode automatic differentiation
- Basic arithmetic: `+`, `-`, `*`, `/`
- Power operation: `pow`
- Topological sorting for backpropagation
- Finite difference gradient verification

## Requirements

A compiler and standard library with C++23 support.

### GCC

```bash
g++ -std=c++23 main.cpp -o main -lstdc++exp
```

### MSVC

```cmd
cl -std:c++latest main.cpp
```


## Implementation

- **TensorImpl**: Stores the value, gradient, previous nodes, and gradient function.
- **Tensor**: Provides data access, gradient access, and backpropagation.
- **Computational Graph**: Records dependencies between Tensor operations.
- **Backpropagation**: Traverses the graph in reverse topological order to calculate gradients.
- **Gradient Verification**: Uses central finite differences to check automatic gradients.

## Notes

- This is an educational project, not a full tensor library.
- Only scalar `float` values are supported.
- Some operations, such as `pow`, have domain restrictions.
- Compiler and standard library support for C++23 features may vary.
