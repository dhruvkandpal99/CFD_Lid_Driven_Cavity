# CFD_LID_DRIVEN_CAVITY

## Overview

The **Lid-Driven Cavity** problem is a classical benchmark in Computational Fluid Dynamics (CFD), used to test and validate numerical solvers and methods. This repository documents my journey into CFD through the implementation and exploration of various approaches to solving this problem.

As I continue to learn and grow in the field of CFD, I will be adding implementations using different numerical schemes, boundary condition treatments, and grid resolutions — starting from simple solvers to more advanced techniques.

## What is the Lid-Driven Cavity Problem?

It involves a square cavity where the top lid moves at a constant velocity while the other three walls remain stationary. Despite its simple geometry, it offers rich flow physics and is widely used to evaluate:

- Numerical accuracy and convergence  
- Stability of CFD solvers  
- Handling of boundary layers and vortices

## Example

![Example Output](https://github.com/user-attachments/assets/cc51f05e-4001-419e-a278-ea0f6ae33e18)

## Goals of this Repository

- Implement and document various CFD methods (e.g., Finite Difference, Finite Volume)  
- Explore different Reynolds numbers and their effects on the flow  
- Validate results with reference benchmarks  
- Visualize and interpret flow patterns and velocity profiles  

## Requirements

To run the simulations, you may need:

- Python 3.x  
- NumPy  
- Matplotlib  
- Jupyter Notebook (optional, for interactive results)  
- tqdm  

### Install the required packages:

```bash
pip install numpy matplotlib tqdm
```

## Status

🚧 Work in Progress — This repository is under active development. Contributions and suggestions are welcome as I continue to improve and expand it.

## License

This project is licensed under the MIT License.
