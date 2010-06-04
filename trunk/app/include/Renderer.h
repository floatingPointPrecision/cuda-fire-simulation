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
@file Renderer.h
@note This file defines the renderer used to combine multiple slices into a single fire image
*/

#pragma once

#include "SliceManager.h"

// Red and Blue are 255 and 0 respectively through all these values so we do not store them
static float colorTempsGreen[12] = {0.2f, 0.27058f, 0.32156f, 0.3647f, 0.4f, 0.43529f, 0.46274f, 0.48627f, 0.5098f, 0.52941f, 0.55294f, 0.57254f};

class Renderer
{
public:
  Renderer(SliceManager* sliceManager, const char* settingsFileName);
  Renderer(const char* settingsFileName);
  ~Renderer();

  void renderSingleSlice(int sliceToDisplay);
  void renderComposite();
  void performMotionBlur();
  void printConstants();

  void increaseTexDensInfluence();
  void increaseTexTempInfluence();
  void increaseDensityAlphaExp();
  void increaseDensityInv();

  void decreaseTexDensInfluence();
  void decreaseTexTempInfluence();
  void decreaseDensityAlphaExp();
  void decreaseDensityInv();

protected:
  void initializeSlabPointers();
  void loadConstantsFromFile(const char* settingsFileName);


  void setupTexture();
  void bindTexture();
  void unbindTexture();
  void deleteTexture();

protected:
  SliceManager* m_sliceManager;

  float** m_d_slabPointers;
  float4* m_d_imageBuffer;
  float4* m_h_imageBuffer;

  float m_maxDensity;
  float m_texDensInfluence;
  float m_texTempInfluence;
  float m_densityAlphaExp;
  float m_sliceSpacing;
  float m_densityInvFactor;

  bool m_isMotionBlurred;

};