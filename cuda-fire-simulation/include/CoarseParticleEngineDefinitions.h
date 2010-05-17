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
@file CoarseParticleEngineDefinitions.h
@brief Defines for the coarse-particle-engine project
*/
#pragma once

namespace cufire
{
  // typedefs
  /**
  * device pointer float4
  */
  typedef thrust::device_ptr<float4> DevPtrFloat4;
  /**
  * float4 iterator for device vectors.
  */
  typedef thrust::device_vector<float4>::iterator DevVecFloat4Itr;
  /**
  * float iterator for device vectors.
  */
  typedef thrust::device_vector<float>::iterator DevVecFloatItr;
  /**
  * Particle iterator tuple. 
  * Tuple containing all the elements of a particle to better coalescing. Elements include:
  * (position.x, position.y, position.z, age), (fuel, radius, mass, impulse), velocity.x, velocity.y, velocity.z
  */
  typedef thrust::tuple<DevPtrFloat4,DevVecFloat4Itr,DevVecFloatItr,DevVecFloatItr,DevVecFloatItr> ParticleItrTuple;
  /**
  * Struct of the iterators which make up a particle.
  * Struct contains all the iterators of a particle to better coalescing. Elements include:
  * (position.x, position.y, position.z, age), (fuel, radius, mass, impulse), velocity.x, velocity.y, velocity.z
  */
  struct ParticleItrStruct
  {
    float4* posAge; ///< (position.x, position.y, position.z, age)
    DevVecFloat4Itr atts; ///< (fuel, radius, mass, impulse)
    DevVecFloatItr velX; ///< velocity.x
    DevVecFloatItr velY; ///< velocity.y
    DevVecFloatItr velZ; ///< velocity.z
  };
  /**
  * Particle tuple. 
  * Tuple containing all the elements of a particle to better coalescing. Elements include:
  * (position.x, position.y, position.z, age), velocity.x, velocity.y, velocity.z, fuel, radius, mass, impulse
  */
  typedef thrust::tuple<float4,float4,float,float,float> ParticleTuple;
  /**
  * Particle iterator.
  */
  typedef thrust::zip_iterator<ParticleItrTuple> ParticleZipItr;


  //#define P_POS 0 ///< position element of a Particle tuple containing 4 floats
  //#define P_X_VEL 1 ///< x velocuity of a Particle tuple
  //#define P_Y_VEL 2 ///< y velocity of a Particle tuple
  //#define P_Z_VEL 3 ///< z velocity of a Particle tuple
  //#define P_FUEL 4 ///< fuel element of a Particle tuple
  //#define P_RADIUS 5 ///< radius element of a Particle tuple
  //#define P_AGE 6 ///< age element of a Particle tuple
  //#define P_MASS 7 ///< mass element of a Particle tuple
  //#define P_IMPULSE 8 ///< impulse element of a Particle tuple

}