# Pendulum Simulation
Here I've implemented a simple rigid-body pendulum simulation from Newton-Euler equations of motion, using explicit Euler integration, rotational damping, and Baumgarte stabilization of the hinge constraints. 1-8 links are combined into a matrix defining their equations of motion and mutual constraints, which is solved using Numpy's `linalg.solve` function.

## Instructions
Requires:
* Python (I use 3.5)
* Numpy
* PyOpenGL
* PyOpenGL-accelerate
* Matplotlib

Run:
`python sim.py [-t|--track]`

Options:
* -t/--track: Enable energy tracking

Controls:
* 'r' : reset the simulation
* '+' : add a link - up to 8
* '-' : remove a link - minimum 1
* 's' : plot energy over time (**requires tracking**)
* 'q/ESC' : quit
