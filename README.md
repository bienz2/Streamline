[![Build Status](https://travis-ci.org/bienz2/Node_Aware_MPI.svg?branch=master)](https://travis-ci.org/bienz2/Node_Aware_MPI)
 
# Node Aware MPI

Node-Aware MPI is a lightweight communication package, which converts standard 
MPI Isends and Irecvs into three step node-aware communication.  This library 
reduces the number and size of messages to be injected into the network.

# Requirements
- `MPI`

# Build Instructions

1. Include headers

# Unit Testing

This package includes a GoogleTest test case in which standard
and node-aware communication results are compared for correctness.

To test:

```bash
mkdir build
cd build
cmake ..
make test
```

# License

This code is distributed under BSD: http://opensource.org/licenses/BSD-2-Clause

Please see `LICENSE.txt` for more information.
