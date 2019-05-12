# Future Prediction in Brownian Dynamics Simulations Using Deep Neural Networks
CS230 Spring 2019 Final Project
Author: Brian K. Ryu

Note: Smaller data set wit

## Requirements
Requirements
```
Tensorflow 1.13.0
NumPy 1.16

```
(Not sure if I will keep this)

## Task
Given an [LAMMPS][LAMMPS] simulation snapshot (.lammpstrj) at a timestep, predict a future snapshot 10<sup>5</sup> timesteps later.

Detailed description can be viewed at the Appendix


## Description of Data Set
[LAMMPS][LAMMPS] (Large-scale Atomic/Molecular Massively Parallel Simulator) is an open-source molecular dynamics software that widely used for scientific research. LAMMPS "trajectory files" (.lammpstrj), which are used as inputs and outputs (labels) in this project, are simulation "snapshots" containing information to a current simulation system. These trajectory files typically contain the following information:
* Number of atoms/particles in simulation box/system
* Dimensions of simulation box
* Current timestep of simulation
* Unique ID and particle type (representing different molecules like O<sub>2</sub> or H<sub>2</sub>O, typically in integer numbers) of each atom/particle in simulation system
* (x,y,z) coordinates of each atom/particle in simulation system
* Linear and angular velocities of each atom/particle
* Charge, dipole moment, and etc. information pertaining to each particle

For this project, I will use minimal trajectory files only containing particle ID's types, and positions (x,y,z). The first several lines look like the following

```
ITEM: TIMESTEP
19400000
ITEM: NUMBER OF ATOMS
8788
ITEM: BOX BOUNDS pp pp pp
0.0000000000000000e+00 3.0024785703928551e+01
0.0000000000000000e+00 3.0024785703928551e+01
0.0000000000000000e+00 3.0024785703928551e+01
ITEM: ATOMS id type xu yu zu 
2578 2 15.7291 30.3692 0.621521 
8139 4 16.7065 0.135016 30.5353 
2122 2 18.7036 0.103409 0.411795 
720 5 25.6909 0.0309724 0.404852 
1346 1 26.5239 30.6178 0.492683 
```

## Preprocessing

[LAMMPS]: https://lammps.sandia.gov/