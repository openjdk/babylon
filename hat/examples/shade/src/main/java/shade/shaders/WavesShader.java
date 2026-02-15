/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package shade.shaders;

import hat.types.F32;
import hat.types.ivec2;
import hat.types.mat3;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import shade.Shader;
import shade.Uniforms;

import static hat.types.F32.pow;
import static hat.types.vec3.clamp;

/*
// afl_ext 2017-2024
// MIT License

// Use your mouse to move the camera around! Press the Left Mouse Button on the image to look around!

#define DRAG_MULT 0.38 // changes how much waves pull on the water
#define WATER_DEPTH 1.0 // how deep is the water
#define CAMERA_HEIGHT 1.5 // how high the camera should be
#define ITERATIONS_RAYMARCH 12 // waves iterations of raymarching
#define ITERATIONS_NORMAL 36 // waves iterations when calculating normals

#define NormalizedMouse (iMouse.xy / iResolution.xy) // normalize mouse coords

// Calculates wave value and its derivative,
// for the wave direction, position in space, wave frequency and time
vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
  float x = dot(direction, position) * frequency + timeshift;
  float wave = exp(sin(x) - 1.0);
  float dx = wave * cos(x);
  return vec2(wave, -dx);
}

// Calculates waves by summing octaves of various waves with various parameters
float getwaves(vec2 position, int iterations) {
  float wavePhaseShift = length(position) * 0.1; // this is to avoid every octave having exactly the same phase everywhere
  float iter = 0.0; // this will help generating well distributed wave directions
  float frequency = 1.0; // frequency of the wave, this will change every iteration
  float timeMultiplier = 2.0; // time multiplier for the wave, this will change every iteration
  float weight = 1.0;// weight in final sum for the wave, this will change every iteration
  float sumOfValues = 0.0; // will store final sum of values
  float sumOfWeights = 0.0; // will store final sum of weights
  for(int i=0; i < iterations; i++) {
    // generate some wave direction that looks kind of random
    vec2 p = vec2(sin(iter), cos(iter));

    // calculate wave data
    vec2 res = wavedx(position, p, frequency, iTime * timeMultiplier + wavePhaseShift);

    // shift position around according to wave drag and derivative of the wave
    position += p * res.y * weight * DRAG_MULT;

    // add the results to sums
    sumOfValues += res.x * weight;
    sumOfWeights += weight;

    // modify next octave ;
    weight = mix(weight, 0.0, 0.2);
    frequency *= 1.18;
    timeMultiplier *= 1.07;

    // add some kind of random value to make next wave look random too
    iter += 1232.399963;
  }
  // calculate and return
  return sumOfValues / sumOfWeights;
}

// Raymarches the ray from top water layer boundary to low water layer boundary
float raymarchwater(vec3 camera, vec3 start, vec3 end, float depth) {
  vec3 pos = start;
  vec3 dir = normalize(end - start);
  for(int i=0; i < 64; i++) {
    // the height is from 0 to -depth
    float height = getwaves(pos.xz, ITERATIONS_RAYMARCH) * depth - depth;
    // if the waves height almost nearly matches the ray height, assume its a hit and return the hit distance
    if(height + 0.01 > pos.y) {
      return distance(pos, camera);
    }
    // iterate forwards according to the height mismatch
    pos += dir * (pos.y - height);
  }
  // if hit was not registered, just assume hit the top layer,
  // this makes the raymarching faster and looks better at higher distances
  return distance(start, camera);
}

// Calculate normal at point by calculating the height at the pos and 2 additional points very close to pos
vec3 normal(vec2 pos, float e, float depth) {
  vec2 ex = vec2(e, 0);
  float H = getwaves(pos.xy, ITERATIONS_NORMAL) * depth;
  vec3 a = vec3(pos.x, H, pos.y);
  return normalize(
    cross(
      a - vec3(pos.x - e, getwaves(pos.xy - ex.xy, ITERATIONS_NORMAL) * depth, pos.y),
      a - vec3(pos.x, getwaves(pos.xy + ex.yx, ITERATIONS_NORMAL) * depth, pos.y + e)
    )
  );
}

// Helper function generating a rotation matrix around the axis by the angle
mat3 createRotationMatrixAxisAngle(vec3 axis, float angle) {
  float s = sin(angle);
  float c = cos(angle);
  float oc = 1.0 - c;
  return mat3(
    oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
    oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
    oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c
  );
}

// Helper function that generates camera ray based on UV and mouse
vec3 getRay(vec2 fragCoord) {
  vec2 uv = ((fragCoord.xy / iResolution.xy) * 2.0 - 1.0) * vec2(iResolution.x / iResolution.y, 1.0);
  // for fisheye, uncomment following line and comment the next one
  //vec3 proj = normalize(vec3(uv.x, uv.y, 1.0) + vec3(uv.x, uv.y, -1.0) * pow(length(uv), 2.0) * 0.05);
  vec3 proj = normalize(vec3(uv.x, uv.y, 1.5));
  if(iResolution.x < 600.0) {
    return proj;
  }
  return createRotationMatrixAxisAngle(vec3(0.0, -1.0, 0.0), 3.0 * ((NormalizedMouse.x + 0.5) * 2.0 - 1.0))
    * createRotationMatrixAxisAngle(vec3(1.0, 0.0, 0.0), 0.5 + 1.5 * (((NormalizedMouse.y == 0.0 ? 0.27 : NormalizedMouse.y) * 1.0) * 2.0 - 1.0))
    * proj;
}

// Ray-Plane intersection checker
float intersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 normal) {
  return clamp(dot(point - origin, normal) / dot(direction, normal), -1.0, 9991999.0);
}

// Some very barebones but fast atmosphere approximation
vec3 extra_cheap_atmosphere(vec3 raydir, vec3 sundir) {
  //sundir.y = max(sundir.y, -0.07);
  float special_trick = 1.0 / (raydir.y * 1.0 + 0.1);
  float special_trick2 = 1.0 / (sundir.y * 11.0 + 1.0);
  float raysundt = pow(abs(dot(sundir, raydir)), 2.0);
  float sundt = pow(max(0.0, dot(sundir, raydir)), 8.0);
  float mymie = sundt * special_trick * 0.2;
  vec3 suncolor = mix(vec3(1.0), max(vec3(0.0), vec3(1.0) - vec3(5.5, 13.0, 22.4) / 22.4), special_trick2);
  vec3 bluesky= vec3(5.5, 13.0, 22.4) / 22.4 * suncolor;
  vec3 bluesky2 = max(vec3(0.0), bluesky - vec3(5.5, 13.0, 22.4) * 0.002 * (special_trick + -6.0 * sundir.y * sundir.y));
  bluesky2 *= special_trick * (0.24 + raysundt * 0.24);
  return bluesky2 * (1.0 + 1.0 * pow(1.0 - raydir.y, 3.0));
}

// Calculate where the sun should be, it will be moving around the sky
vec3 getSunDirection() {
  return normalize(vec3(-0.0773502691896258 , 0.5 + sin(iTime * 0.2 + 2.6) * 0.45 , 0.5773502691896258));
}

// Get atmosphere color for given direction
vec3 getAtmosphere(vec3 dir) {
   return extra_cheap_atmosphere(dir, getSunDirection()) * 0.5;
}

// Get sun color for given direction
float getSun(vec3 dir) {
  return pow(max(0.0, dot(dir, getSunDirection())), 720.0) * 210.0;
}

// Great tonemapping function from my other shader: https://www.shadertoy.com/view/XsGfWV
vec3 aces_tonemap(vec3 color) {
  mat3 m1 = mat3(
    0.59719, 0.07600, 0.02840,
    0.35458, 0.90834, 0.13383,
    0.04823, 0.01566, 0.83777
  );
  mat3 m2 = mat3(
    1.60475, -0.10208, -0.00327,
    -0.53108,  1.10813, -0.07276,
    -0.07367, -0.00605,  1.07602
  );
  vec3 v = m1 * color;
  vec3 a = v * (v + 0.0245786) - 0.000090537;
  vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}

// Main
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  // get the ray
  vec3 ray = getRay(fragCoord);
  if(ray.y >= 0.0) {
    // if ray.y is positive, render the sky
    vec3 C = getAtmosphere(ray) + getSun(ray);
    fragColor = vec4(aces_tonemap(C * 2.0),1.0);
    return;
  }

  // now ray.y must be negative, water must be hit
  // define water planes
  vec3 waterPlaneHigh = vec3(0.0, 0.0, 0.0);
  vec3 waterPlaneLow = vec3(0.0, -WATER_DEPTH, 0.0);

  // define ray origin, moving around
  vec3 origin = vec3(iTime * 0.2, CAMERA_HEIGHT, 1);

  // calculate intersections and reconstruct positions
  float highPlaneHit = intersectPlane(origin, ray, waterPlaneHigh, vec3(0.0, 1.0, 0.0));
  float lowPlaneHit = intersectPlane(origin, ray, waterPlaneLow, vec3(0.0, 1.0, 0.0));
  vec3 highHitPos = origin + ray * highPlaneHit;
  vec3 lowHitPos = origin + ray * lowPlaneHit;

  // raymatch water and reconstruct the hit pos
  float dist = raymarchwater(origin, highHitPos, lowHitPos, WATER_DEPTH);
  vec3 waterHitPos = origin + ray * dist;

  // calculate normal at the hit position
  vec3 N = normal(waterHitPos.xz, 0.01, WATER_DEPTH);

  // smooth the normal with distance to avoid disturbing high frequency noise
  N = mix(N, vec3(0.0, 1.0, 0.0), 0.8 * min(1.0, sqrt(dist*0.01) * 1.1));

  // calculate fresnel coefficient
  float fresnel = (0.04 + (1.0-0.04)*(pow(1.0 - max(0.0, dot(-N, ray)), 5.0)));

  // reflect the ray and make sure it bounces up
  vec3 R = normalize(reflect(ray, N));
  R.y = abs(R.y);

  // calculate the reflection and approximate subsurface scattering
  vec3 reflection = getAtmosphere(R) + getSun(R);
  vec3 scattering = vec3(0.0293, 0.0698, 0.1717) * 0.1 * (0.2 + (waterHitPos.y + WATER_DEPTH) / WATER_DEPTH);

  // return the combined result
  vec3 C = fresnel * reflection + scattering;
  fragColor = vec4(aces_tonemap(C * 2.0), 1.0);
}
*/

//https://www.shadertoy.com/view/MdXyzX
public class WavesShader implements Shader {
    public static String glslSource = """

                              #define DRAG_MULT 0.38 // changes how much waves pull on the water
                              #define WATER_DEPTH 1.0 // how deep is the water
                              #define CAMERA_HEIGHT 1.5 // how high the camera should be
                              #define ITERATIONS_RAYMARCH 12 // waves iterations of raymarching
                              #define ITERATIONS_NORMAL 36 // waves iterations when calculating normals

                              #define NormalizedMouse (iMouse.xy / iResolution.xy) // normalize mouse coords

                              // Calculates wave value and its derivative,
                              // for the wave direction, position in space, wave frequency and time
                              vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
                                float x = dot(direction, position) * frequency + timeshift;
                                float wave = exp(sin(x) - 1.0);
                                float dx = wave * cos(x);
                                return vec2(wave, -dx);
                              }

                              // Calculates waves by summing octaves of various waves with various parameters
                              float getwaves(vec2 position, int iterations) {
                                float wavePhaseShift = length(position) * 0.1; // this is to avoid every octave having exactly the same phase everywhere
                                float iter = 0.0; // this will help generating well distributed wave directions
                                float frequency = 1.0; // frequency of the wave, this will change every iteration
                                float timeMultiplier = 2.0; // time multiplier for the wave, this will change every iteration
                                float weight = 1.0;// weight in final sum for the wave, this will change every iteration
                                float sumOfValues = 0.0; // will store final sum of values
                                float sumOfWeights = 0.0; // will store final sum of weights
                                for(int i=0; i < iterations; i++) {
                                  // generate some wave direction that looks kind of random
                                  vec2 p = vec2(sin(iter), cos(iter));

                                  // calculate wave data
                                  vec2 res = wavedx(position, p, frequency, iTime * timeMultiplier + wavePhaseShift);

                                  // shift position around according to wave drag and derivative of the wave
                                  position += p * res.y * weight * DRAG_MULT;

                                  // add the results to sums
                                  sumOfValues += res.x * weight;
                                  sumOfWeights += weight;

                                  // modify next octave ;
                                  weight = mix(weight, 0.0, 0.2);
                                  frequency *= 1.18;
                                  timeMultiplier *= 1.07;

                                  // add some kind of random value to make next wave look random too
                                  iter += 1232.399963;
                                }
                                // calculate and return
                                return sumOfValues / sumOfWeights;
                              }

                              // Raymarches the ray from top water layer boundary to low water layer boundary
                              float raymarchwater(vec3 camera, vec3 start, vec3 end, float depth) {
                                vec3 pos = start;
                                vec3 dir = normalize(end - start);
                                for(int i=0; i < 64; i++) {
                                  // the height is from 0 to -depth
                                  float height = getwaves(pos.xz, ITERATIONS_RAYMARCH) * depth - depth;
                                  // if the waves height almost nearly matches the ray height, assume its a hit and return the hit distance
                                  if(height + 0.01 > pos.y) {
                                    return distance(pos, camera);
                                  }
                                  // iterate forwards according to the height mismatch
                                  pos += dir * (pos.y - height);
                                }
                                // if hit was not registered, just assume hit the top layer,
                                // this makes the raymarching faster and looks better at higher distances
                                return distance(start, camera);
                              }

                              // Calculate normal at point by calculating the height at the pos and 2 additional points very close to pos
                              vec3 normal(vec2 pos, float e, float depth) {
                                vec2 ex = vec2(e, 0);
                                float H = getwaves(pos.xy, ITERATIONS_NORMAL) * depth;
                                vec3 a = vec3(pos.x, H, pos.y);
                                return normalize(
                                  cross(
                                    a - vec3(pos.x - e, getwaves(pos.xy - ex.xy, ITERATIONS_NORMAL) * depth, pos.y),
                                    a - vec3(pos.x, getwaves(pos.xy + ex.yx, ITERATIONS_NORMAL) * depth, pos.y + e)
                                  )
                                );
                              }

                              // Helper function generating a rotation matrix around the axis by the angle
                              mat3 createRotationMatrixAxisAngle(vec3 axis, float angle) {
                                float s = sin(angle);
                                float c = cos(angle);
                                float oc = 1.0 - c;
                                return mat3(
                                  oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
                                  oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
                                  oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c
                                );
                              }

                              // Helper function that generates camera ray based on UV and mouse
                              vec3 getRay(vec2 fragCoord) {
                                vec2 uv = ((fragCoord.xy / iResolution.xy) * 2.0 - 1.0) * vec2(iResolution.x / iResolution.y, 1.0);
                                // for fisheye, uncomment following line and comment the next one
                                //vec3 proj = normalize(vec3(uv.x, uv.y, 1.0) + vec3(uv.x, uv.y, -1.0) * pow(length(uv), 2.0) * 0.05);
                                vec3 proj = normalize(vec3(uv.x, uv.y, 1.5));
                                if(iResolution.x < 600.0) {
                                  return proj;
                                }
                                return createRotationMatrixAxisAngle(vec3(0.0, -1.0, 0.0), 3.0 * ((NormalizedMouse.x + 0.5) * 2.0 - 1.0))
                                  * createRotationMatrixAxisAngle(vec3(1.0, 0.0, 0.0), 0.5 + 1.5 * (((NormalizedMouse.y == 0.0 ? 0.27 : NormalizedMouse.y) * 1.0) * 2.0 - 1.0))
                                  * proj;
                              }

                              // Ray-Plane intersection checker
                              float intersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 normal) {
                                return clamp(dot(point - origin, normal) / dot(direction, normal), -1.0, 9991999.0);
                              }

                              // Some very barebones but fast atmosphere approximation
                              vec3 extra_cheap_atmosphere(vec3 raydir, vec3 sundir) {
                                //sundir.y = max(sundir.y, -0.07);
                                float special_trick = 1.0 / (raydir.y * 1.0 + 0.1);
                                float special_trick2 = 1.0 / (sundir.y * 11.0 + 1.0);
                                float raysundt = pow(abs(dot(sundir, raydir)), 2.0);
                                float sundt = pow(max(0.0, dot(sundir, raydir)), 8.0);
                                float mymie = sundt * special_trick * 0.2;
                                vec3 suncolor = mix(vec3(1.0), max(vec3(0.0), vec3(1.0) - vec3(5.5, 13.0, 22.4) / 22.4), special_trick2);
                                vec3 bluesky= vec3(5.5, 13.0, 22.4) / 22.4 * suncolor;
                                vec3 bluesky2 = max(vec3(0.0), bluesky - vec3(5.5, 13.0, 22.4) * 0.002 * (special_trick + -6.0 * sundir.y * sundir.y));
                                bluesky2 *= special_trick * (0.24 + raysundt * 0.24);
                                return bluesky2 * (1.0 + 1.0 * pow(1.0 - raydir.y, 3.0));
                              }

                              // Calculate where the sun should be, it will be moving around the sky
                              vec3 getSunDirection() {
                                return normalize(vec3(-0.0773502691896258 , 0.5 + sin(iTime * 0.2 + 2.6) * 0.45 , 0.5773502691896258));
                              }

                              // Get atmosphere color for given direction
                              vec3 getAtmosphere(vec3 dir) {
                                 return extra_cheap_atmosphere(dir, getSunDirection()) * 0.5;
                              }

                              // Get sun color for given direction
                              float getSun(vec3 dir) {
                                return pow(max(0.0, dot(dir, getSunDirection())), 720.0) * 210.0;
                              }

                              // Great tonemapping function from my other shader: https://www.shadertoy.com/view/XsGfWV
                              vec3 aces_tonemap(vec3 color) {
                                mat3 m1 = mat3(
                                  0.59719, 0.07600, 0.02840,
                                  0.35458, 0.90834, 0.13383,
                                  0.04823, 0.01566, 0.83777
                                );
                                mat3 m2 = mat3(
                                  1.60475, -0.10208, -0.00327,
                                  -0.53108,  1.10813, -0.07276,
                                  -0.07367, -0.00605,  1.07602
                                );
                                vec3 v = m1 * color;
                                vec3 a = v * (v + 0.0245786) - 0.000090537;
                                vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
                                return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
                              }

                              // Main
                              void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                                // get the ray
                                vec3 ray = getRay(fragCoord);
                                if(ray.y >= 0.0) {
                                  // if ray.y is positive, render the sky
                                  vec3 C = getAtmosphere(ray) + getSun(ray);
                                  fragColor = vec4(aces_tonemap(C * 2.0),1.0);
                                  return;
                                }

                                // now ray.y must be negative, water must be hit
                                // define water planes
                                vec3 waterPlaneHigh = vec3(0.0, 0.0, 0.0);
                                vec3 waterPlaneLow = vec3(0.0, -WATER_DEPTH, 0.0);

                                // define ray origin, moving around
                                vec3 origin = vec3(iTime * 0.2, CAMERA_HEIGHT, 1);

                                // calculate intersections and reconstruct positions
                                float highPlaneHit = intersectPlane(origin, ray, waterPlaneHigh, vec3(0.0, 1.0, 0.0));
                                float lowPlaneHit = intersectPlane(origin, ray, waterPlaneLow, vec3(0.0, 1.0, 0.0));
                                vec3 highHitPos = origin + ray * highPlaneHit;
                                vec3 lowHitPos = origin + ray * lowPlaneHit;

                                // raymatch water and reconstruct the hit pos
                                float dist = raymarchwater(origin, highHitPos, lowHitPos, WATER_DEPTH);
                                vec3 waterHitPos = origin + ray * dist;

                                // calculate normal at the hit position
                                vec3 N = normal(waterHitPos.xz, 0.01, WATER_DEPTH);

                                // smooth the normal with distance to avoid disturbing high frequency noise
                                N = mix(N, vec3(0.0, 1.0, 0.0), 0.8 * min(1.0, sqrt(dist*0.01) * 1.1));

                                // calculate fresnel coefficient
                                float fresnel = (0.04 + (1.0-0.04)*(pow(1.0 - max(0.0, dot(-N, ray)), 5.0)));

                                // reflect the ray and make sure it bounces up
                                vec3 R = normalize(reflect(ray, N));
                                R.y = abs(R.y);

                                // calculate the reflection and approximate subsurface scattering
                                vec3 reflection = getAtmosphere(R) + getSun(R);
                                vec3 scattering = vec3(0.0293, 0.0698, 0.1717) * 0.1 * (0.2 + (waterHitPos.y + WATER_DEPTH) / WATER_DEPTH);

                                // return the combined result
                                vec3 C = fresnel * reflection + scattering;
                                fragColor = vec4(aces_tonemap(C * 2.0), 1.0);
                              }
                """;
    static float DRAG_MULT = 0.38f; // changes how much waves pull on the water
    static float WATER_DEPTH = 1f; // how deep is the water
    static float CAMERA_HEIGHT = 1.5f; // how high the camera should be
    static int ITERATIONS_RAYMARCH = 12; // waves iterations of raymarching
    static int ITERATIONS_NORMAL = 36; // waves iterations when calculating normals

    static vec2 NormalizedMouse(vec2 fMouse, vec2 fResolution) {
        return vec2.div(fMouse,fResolution);
    } // normalize mouse coords

    // Calculates wave value and its derivative,
// for the wave direction, position in space, wave frequency and time
    static vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
        float x = vec2.dot(direction,position) * frequency + timeshift;
        float wave = F32.exp(F32.sin(x) - 1.0f);
        float dx = wave * F32.cos(x);
        return vec2.vec2(wave, -dx);
    }

    // Calculates waves by summing octaves of various waves with various parameters
    static float getwaves(vec2 position, int iterations, float fTime) {
        float wavePhaseShift = vec2.length(position) * 0.1f; // this is to avoid every octave having exactly the same phase everywhere
        float iter = 0.0f; // this will help generating well distributed wave directions
        float frequency = 1.0f; // frequency of the wave, this will change every iteration
        float timeMultiplier = 2.0f; // time multiplier for the wave, this will change every iteration
        float weight = 1.0f;// weight in final sum for the wave, this will change every iteration
        float sumOfValues = 0.0f; // will store final sum of values
        float sumOfWeights = 0.0f; // will store final sum of weights
        for (int i = 0; i < iterations; i++) {
            // generate some wave direction that looks kind of random
            vec2 p = vec2.vec2(F32.sin(iter), F32.cos(iter));

            // calculate wave data
            vec2 res = wavedx(position, p, frequency, fTime * timeMultiplier + wavePhaseShift);

            // shift position around according to wave drag and derivative of the wave
            position = vec2.add(position, vec2.mul(position, res.y() * weight * DRAG_MULT));

            // add the results to sums
            sumOfValues += res.x() * weight;
            sumOfWeights += weight;

            // modify next octave ;
            weight = F32.mix(weight, 0.0f, 0.2f);
            frequency *= 1.18f;
            timeMultiplier *= 1.07f;

            // add some kind of random value to make next wave look random too
            iter += 1232.399963f;
        }
        // calculate and return
        return sumOfValues / sumOfWeights;
    }

    // Raymarches the ray from top water layer boundary to low water layer boundary
    static float raymarchwater(vec3 camera, vec3 start, vec3 end, float depth, float fTime) {
        vec3 pos = start;
        vec3 dir = end.sub(start).normalize();
        for (int i = 0; i < 64; i++) {
            // the height is from 0 to -depth
            float height = getwaves(vec2.vec2(pos.x(), pos.y()), ITERATIONS_RAYMARCH, fTime) * depth - depth;
            // if the waves height almost nearly matches the ray height, assume its a hit and return the hit distance
            if (height + 0.01f > pos.y()) {
                return pos.distance(camera);
            }
            // iterate forwards according to the height mismatch
            pos = pos.add(dir.mul(pos.y() - height));
        }
        // if hit was not registered, just assume hit the top layer,
        // this makes the raymarching faster and looks better at higher distances
        return start.distance(camera);
    }

    // Calculate normal at point by calculating the height at the pos and 2 additional points very close to pos
    static vec3 normal(vec2 pos, float e, float depth, float fTime) {
        vec2 ex = vec2.vec2(e, 0);
        float H = getwaves(pos, ITERATIONS_NORMAL, fTime) * depth;
        vec3 a = vec3.vec3(pos.x(), H, pos.y());
        return vec3.normalize(
                vec3.cross(
                        a.sub(vec3.vec3(pos.x() - e, getwaves(vec2.sub(pos,ex), ITERATIONS_NORMAL, fTime) * depth, pos.y())),
                        a.sub(vec3.vec3(pos.x(), getwaves(vec2.add(pos, ex), ITERATIONS_NORMAL, fTime) * depth, pos.y() + e))
                )
        );
    }

    // Helper function generating a rotation matrix around the axis by the angle
    static mat3 createRotationMatrixAxisAngle(vec3 axis, float angle) {
        float s = F32.sin(angle);
        float c = F32.cos(angle);
        float oc = 1.0f - c;
        return mat3.mat3(
                oc * axis.x() * axis.x() + c, oc * axis.x() * axis.y() - axis.z() * s, oc * axis.z() * axis.x() + axis.y() * s,
                oc * axis.x() * axis.y() + axis.z() * s, oc * axis.y() * axis.y() + c, oc * axis.y() * axis.z() - axis.x() * s,
                oc * axis.z() * axis.x() - axis.y() * s, oc * axis.y() * axis.z() + axis.x() * s, oc * axis.z() * axis.z() + c
        );
    }

    // Helper function that generates camera ray based on UV and mouse
    static vec3 getRay(vec2 fragCoord, vec2 fres, vec2 fMouse) {
        vec2 uv = vec2.mul(vec2.div(fragCoord,vec2.sub(vec2.mul(fres, 2f),1f)),vec2.vec2(fres.x() / fres.y(), 1.0f));
        // for fisheye, uncomment following line and comment the next one
        //vec3 proj = normalize(vec3(uv.x, uv.y, 1.0) + vec3(uv.x, uv.y, -1.0) * pow(length(uv), 2.0) * 0.05);
        vec3 proj = vec3.vec3(uv.x(), uv.y(), 1.5f).normalize();
        if (fres.x() < 600.0) {
            return proj;
        }

        var m1 = createRotationMatrixAxisAngle(vec3.vec3(0.0f, -1.0f, 0.0f), 3.0f * ((NormalizedMouse(fMouse, fres).x() + 0.5f) * 2.0f - 1.0f));
        var m2 = createRotationMatrixAxisAngle(vec3.vec3(1.0f, 0.0f, 0.0f), 0.5f + 1.5f * (((NormalizedMouse(fMouse, fres).y()) == 0.0f ? 0.27f : NormalizedMouse(fMouse, fres).y() * 1.0f) * 2.0f - 1.0f));
        return proj.mul(m1).mul(m2);
    }

    // Ray-Plane intersection checker
    static float intersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 normal) {
        return F32.clamp(vec3.dot(point.sub(origin), normal) / vec3.dot(direction, normal), -1.0f, 9991999.0f);
    }

    // Some very barebones but fast atmosphere approximation
    static vec3 extra_cheap_atmosphere(vec3 raydir, vec3 sundir, float fTime) {
        //sundir.y = max(sundir.y, -0.07);
        float special_trick = 1.0f / (raydir.y() * 1.0f + 0.1f);
        float special_trick2 = 1.0f / (sundir.y() * 11.0f + 1.0f);
        float raysundt = F32.pow(F32.abs(vec3.dot(sundir, raydir)), 2.0f);
        float sundt = F32.pow(F32.max(0.0f, vec3.dot(sundir, raydir)), 8.0f);
        float mymie = sundt * special_trick * 0.2f;
        vec3 suncolor = vec3.vec3(0f, 0f, .7f);
        // vec3 suncolor = vec3.mix(
        //       vec3(1.0f), vec3.max(vec3(0.0f), vec3(1.0f).sub(vec3(5.5f, 13.0f, 22.4f).div(22.4f)
        //     ), special_trick2));
        vec3 bluesky = vec3.vec3(5.5f, 13.0f, 22.4f).div(22.4f).mul(suncolor);
        vec3 bluesky2 = vec3.max(vec3.vec3(0.0f), bluesky.sub(vec3.vec3(5.5f, 13.0f, 22.4f).mul(0.002f * (special_trick + -6.0f * sundir.y() * sundir.y()))));
        bluesky2 = bluesky2.mul(special_trick * (0.24f + raysundt * 0.24f));
        return bluesky2.mul(1.0f + F32.pow(1.0f - raydir.y(), 3.0f));
    }

    // Calculate where the sun should be, it will be moving around the sky
    static vec3 getSunDirection(float fTime) {
        return vec3.vec3(-0.0773502691896258f, 0.5f + F32.sin(fTime * 0.2f + 2.6f) * 0.45f, 0.5773502691896258f).normalize();
    }

    // Get atmosphere color for given direction
    static vec3 getAtmosphere(vec3 dir, float fTime) {
        return extra_cheap_atmosphere(dir, getSunDirection(fTime), fTime).mul(0.5f);
    }

    // Get sun color for given direction
    static float getSun(vec3 dir, float fTime) {
        return F32.pow(F32.max(0.0f, vec3.dot(dir, getSunDirection(fTime))), 720.0f) * 210.0f;
    }

    static vec3 aces_tonemap(vec3 color) {
        mat3 m1 = mat3.mat3(
                0.59719f, 0.07600f, 0.02840f,
                0.35458f, 0.90834f, 0.13383f,
                0.04823f, 0.01566f, 0.83777f
        );
        mat3 m2 = mat3.mat3(
                1.60475f, -0.10208f, -0.00327f,
                -0.53108f, 1.10813f, -0.07276f,
                -0.07367f, -0.00605f, 1.07602f
        );
        vec3 v = color.mul(m1);
        vec3 a = v.mul(v.add(+0.0245786f)).sub(0.000090537f);
        vec3 b = v.mul((v.mul(0.983729f).add(0.4329510f)).add(0.238081f));
        var aOverBMulM2 = a.div(b).mul(m2);
        return vec3.clamp(vec3.pow(aOverBMulM2, 1.0f / 2.2f), 0f, 1f);
    }

    //https://www.shadertoy.com/view/MdXyzX
    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        var fResolution = vec2.vec2(uniforms.iResolution());
        var fMouse = vec2.vec2(uniforms.iMouse());
        float fTime = uniforms.iTime();
        // get the ray
        vec3 ray = getRay(fragCoord, fResolution, fMouse);

        if (ray.y() >= 0.0) {
            // if ray.y is positive, render the sky
            vec3 C = getAtmosphere(ray, fTime).add(getSun(ray, fTime));
            fragColor = vec4.vec4(aces_tonemap(C.mul(2.0f)), 1.0f);
            return fragColor;
        } else {

            // now ray.y must be negative, water must be hit
            // define water planes
            vec3 waterPlaneHigh = vec3.vec3(0.0f, 0.0f, 0.0f);
            vec3 waterPlaneLow = vec3.vec3(0.0f, -WATER_DEPTH, 0.0f);

            // define ray origin, moving around
            vec3 origin = vec3.vec3(fTime * 0.2f, CAMERA_HEIGHT, 1f);

            // calculate intersections and reconstruct positions
            float highPlaneHit = intersectPlane(origin, ray, waterPlaneHigh, vec3.vec3(0.0f, 1.0f, 0.0f));
            float lowPlaneHit = intersectPlane(origin, ray, waterPlaneLow, vec3.vec3(0.0f, 1.0f, 0.0f));
            vec3 highHitPos = origin.add(ray.mul(highPlaneHit));
            vec3 lowHitPos = origin.add(ray.mul(lowPlaneHit));

            // raymatch water and reconstruct the hit pos
            float dist = raymarchwater(origin, highHitPos, lowHitPos, WATER_DEPTH, fTime);
            vec3 waterHitPos = origin.add(ray.mul(dist));

            // calculate normal at the hit position
            vec3 N = normal(vec2.vec2(waterHitPos.x(), waterHitPos.y()), 0.01f, WATER_DEPTH, fTime);

            // smooth the normal with distance to avoid disturbing high frequency noise
            N = vec3.mix(N, vec3.vec3(0.0f, 1.0f, 0.0f), 0.8f * F32.min(1.0f, F32.sqrt(dist * 0.01f) * 1.1f));

            // calculate fresnel coefficient
            float fresnel = (0.04f + (1.0f - 0.04f) * (pow(1.0f - F32.max(0.0f, vec3.dot(vec3.vec3(0f).sub(N), ray)), 5.0f)));

            // reflect the ray and make sure it bounces up
            vec3 R = vec3.normalize(vec3.reflect(ray, N));
            R = vec3.vec3(R.x(), F32.abs(R.y()), R.z());

            // calculate the reflection and approximate subsurface scattering
            vec3 reflection = getAtmosphere(R, fTime).add(getSun(R, fTime));
            vec3 scattering = vec3.vec3(0.0293f, 0.0698f, 0.1717f).mul(0.1f).mul(0.2f + (waterHitPos.y() + WATER_DEPTH) / WATER_DEPTH);

            // return the combined result
            vec3 C = reflection.mul(fresnel).add(scattering);
            fragColor = vec4.vec4(clamp(aces_tonemap(C.mul(2.0f)), 0f, 1f), 1.0f);
            return fragColor;
        }
    };

}
