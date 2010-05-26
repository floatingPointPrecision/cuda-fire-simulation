/*******************************************************************************
Copyright (c) 2010, Steve Lesser
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1) Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2)Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3) The name of contributors may not be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL STEVE LESSER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

/**
@file DoxyMainPage.h
@brief Specifies the main page documentation for doxygen
*/

/**
\mainpage

\section Introduction
This application is meant to simulate fire on the GPU using CUDA. It was originally built by Steve Lesser for the Stanford courses CS 348B Image Synthesis,
CS 193G Programming Massively Parallel Processors, and CS 448X The Math and Computer Science of Hollywood Special Effects.
\n\n
The techniques used in this program are based off the 2009 SIGGRAPH paper Directable, High-Resolution Simulation of Fire on the GPU by Christopher Horvath
and Willi Geiger.
\n\n
The original project website for this application can be found at http://code.google.com/p/cuda-fire-simulation/

\section Requirements
cuda-fire-simulation was built on the following platform and until further notice should require the following as well:
\li Windows 64-bit
\li Microsoft Visual Studio 2008
\li CUDA-enabled device
\li 64-bit CUDA driver (http://developer.nvidia.com/object/cuda_3_0_downloads.html)
\li CUDA toolkit (http://developer.nvidia.com/object/cuda_3_0_downloads.html)
\li CUDA SDK (http://developer.nvidia.com/object/cuda_3_0_downloads.html)
\li CUDA VS Wizard (http://forums.nvidia.com/index.php?showtopic=83054)

\section Overview
The simulation is broken up into a few pieces including:
\li Fire application. The entry point of the application and connects the components together
\li Coarse particle engine. This is used to define the broad movement of the fire and is meant to be fairly customizable
\li Grid refinement. This is used to perform the 2D fire / fluid simulations in the grid refinement stage
\li XML parser. A simple utility to easily read in XML files and quickly extract basic values.
\n\n
Except for Fire Application which creates the application, each of these components is a separate Visual Studio project which exports a library.
All the projects can be found in the cuda-fire-simulation\cuda-fire-simulation.sln Visual Studio solution

\section ExternalSources
This project takes advantage of several open source libraries which are included with the project and do not need to be downloaded separately:
\li Thrust. Thrust is a CUDA library which allows vector operations and optimized common algorithms. http://code.google.com/p/thrust/
\li OpenCurrent. OpenCurrent is a CUDA library for very fast partial differential equation solvers. http://code.google.com/p/opencurrent/
\li TinyXML. TinyXML is a very lightweight XML reader used to create the XMLParser class. http://www.grinninglizard.com/tinyxml/
*/