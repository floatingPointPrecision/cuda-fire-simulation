# Objective #

Use NVIDIA's CUDA architecture to implement a fire simulation application based on the 2009 SIGGRAPH paper [Directable, High-Resolution Simulation of Fire on the GPU](http://portal.acm.org/citation.cfm?id=1531326.1531347) by Christopher Horvath and Willi Geiger.

# Completion #

Note that this project is still in active development so functionality is not yet complete. If interested in trying out the program I would check back around June 15th after all my projects are due and the whole application will (hopefully) be complete.

Currently, the application performs the 3D coarse particle fluid simulation and 2D fluid simulation for each slice. The user-defined 3D force field used to influence the particle movement is currently controlled by manipulating kernels but an easier method is to come. A few constant values are read in by a very simple XML class reader I wrote which takes advantage of [tinyXML](http://www.grinninglizard.com/tinyxml/), soon more of the program will be controlled through this XML file.

The user can move around the coarse 3D simulation and can switch the visualization between the 3D simulation and each of the 2D slices (temperature, texture, fuel, density, and velocity) and can move between different slices. The user can also pause and resume the current simulation.

I'm currently working on rendering and controlling the birth and death of particles.

I'm trying to come up with a guide to setting up [OpenCurrent](http://code.google.com/p/opencurrent) for Windows 7, since it involved changing quite a bit of the OpenCurrent download to work.

# Requirements #

This application requires [CUDA](http://developer.nvidia.com/object/cuda_3_0_downloads.html), [CUDA SDK](http://developer.nvidia.com/object/cuda_3_0_downloads.html), [GLUT](http://www.xmission.com/~nate/glut.html), and Jonathon Cohen's CUDA PDE Solver [OpenCurrent](http://code.google.com/p/opencurrent/). The [Thrust](http://code.google.com/p/thrust/) library is used as well and is included in the include folder so it does not need to be separately installed. In it's current state I've only tested it on my machine which is Windows 7 64-bit and Visual Studio 2008, but other ports will come.

# Background #

This simulation aspect of this project was created for the final project for CS 193G Programming Massively Parallel Processors with CUDA. The rendering aspect was created for the final project for CS 348B Image Synthesis.

# Contact Information #

Steve Lesser:

e-mail: sklesser (at) gmail.com