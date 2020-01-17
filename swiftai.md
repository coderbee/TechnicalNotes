# Swiftai Notes

#### Metaprogramming 
Metaprogramming is a programming technique in which computer programs have the ability to treat other programs as their data.
[Source](https://docs.google.com/document/d/1UIPWl4lvBTozBD5OQ9SrxgcM7rA4pODMOjqQv3tm57w/edit?usp=sharing)
#### **Quoting** 
provides a notation for embedding languages into one another. Just like double quotes allow programmers to embed string literals, there are notations that generalize this notion to embedding code literals and DSLs.
[Source](https://docs.google.com/document/d/1UIPWl4lvBTozBD5OQ9SrxgcM7rA4pODMOjqQv3tm57w/edit?usp=sharing)
#### Quasiquoting 
extends quoting by introducing means to insert things from the enclosing language into quotes (also known as unquoting or splicing). Just like string interpolation extends string literals, quasiquotes extend quotes. Some notations can also be extended to not only insert but to also extract things into the enclosing language when used in pattern matching. Pioneered by Lisp, quasiquoting has found a home in many programming languages.
[Source](https://docs.google.com/document/d/1UIPWl4lvBTozBD5OQ9SrxgcM7rA4pODMOjqQv3tm57w/edit?usp=sharing)
#### Multi-Level Intermediate Representation (MLIR) 
is a generalization of both the LLVM IR and TensorFlow graphs to represent arbitrary computations at multiple levels of abstraction to enable domain-specific optimizations and code generation (e.g. for CPUs, GPUs, TPUs, and other hardware targets).
[Source](https://docs.google.com/document/d/1UIPWl4lvBTozBD5OQ9SrxgcM7rA4pODMOjqQv3tm57w/edit?usp=sharing)
#### Polyhedral Loop Optimization
  - High-Dimensional Loop Nests
  - Multi-Dimensional Arrays
  
  Maps, sets, and relations with affine constraints are the core structures underlying a polyhedral representation of high-dimensional loop nests and multi-dimensional arrays.

#### Dialect 

#### Stencil 
#### Generics
#### Closure

### References
- [MLIR Specification](https://github.com/tensorflow/mlir/blob/408b626c1598e31dc31abae55131c8a17063a2a2/g3doc/LangRef.md#high-level-structure)
- [MLIR: The case for a simplified polyhedral form](https://github.com/tensorflow/mlir/blob/408b626c1598e31dc31abae55131c8a17063a2a2/g3doc/RationaleSimplifiedPolyhedralForm.md#mlir-the-case-for-a-simplified-polyhedral-form)
- [MLIR Rationale](https://github.com/tensorflow/mlir/blob/408b626c1598e31dc31abae55131c8a17063a2a2/g3doc/Rationale.md)
- [Polyhedral Compilation as Design Pattern for Compilers-Youtube](https://www.youtube.com/watch?v=mt6pIpt5Wk0)
# DeepLearningCourse
learnings from Coursera's Deep learning course .  

## The broad steps in implementing neural netwoks are as follows

1) Define Model Structure 
    - number of layers, # neurons per layer
    - activation functions to use
2) Initialize model parameters
3) Iterate through this
    - Calculate cost function J using forward propogation
    - Calculate the current gradients using backward propogation
    - Update weights 

## Common Python commands used
```
import numpy as np                      => numpy is a great choice for linear algebra/math functions needed for NN  
m_train = train_set_x_orig.shape[0]     => use .shape, .reshape to access dimensions, vectorize matrices into columns
np.zeros([dim,1])                       => initialize matrices with 0's
dw = np.dot(X, (A - Y).T) /m            => dot product equivalent to Matrix multiplication ( 2D matrices)
c1 = np.multiply(Y, np.log(A))          => Element-wise matrix multiplication
db = np.sum(A - Y ) /m                  => Subtract matrices A - Y (broadcasting if needed), then add element-wise to give a single number
```
