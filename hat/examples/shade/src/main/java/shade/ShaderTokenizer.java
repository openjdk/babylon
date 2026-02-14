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
package shade;

import optkl.textmodel.terminal.ANSI;
import optkl.util.Regex;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;

public class ShaderTokenizer {
    interface Token {
        TOKEN_TYPE tokenType();
        String value();

    }
    interface ParentToken extends Token {
        List<Token> children();
    }
    interface LeafToken extends Token {

    }

    interface TypeToken extends LeafToken{}
    interface ConstToken extends Token{}

    record CreateToken(TOKEN_TYPE tokenType, String value) implements ConstToken {
        public String toString() {return tokenType + ":" + value;}
    }
    record MathLibCallToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {return tokenType + ":" + value;}
    }
    record CallToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {return tokenType + ":" + value;}
    }
    record VecTypeToken(TOKEN_TYPE tokenType, String value) implements TypeToken {
        public String toString() {return tokenType + ":" + value;}
    }
    record MatTypeToken(TOKEN_TYPE tokenType, String value) implements TypeToken {
        public String toString() {return tokenType + ":" + value;}
    }
    record PrimitiveTypeToken(TOKEN_TYPE tokenType, String value) implements TypeToken {
        public String toString() {return tokenType + ":" + value;}
    }
    record FloatToken(TOKEN_TYPE tokenType, String value, float f32) implements ConstToken {
        public String toString() {return "F32:"+f32;}
    }
    record WhiteSpaceToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {return "WS:" + value.replace("\n", "\\n").replace("\t","\\t").replace(" ", ".");}
    }
    record IntToken(TOKEN_TYPE tokenType, String value, int i32) implements ConstToken {
        public String toString() {return "S32:"+i32;}
    }
    record ReservedToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {return tokenType + ":" + value;}
    }
    record UniformToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {return tokenType + ":" + value;}
    }
    record SimpleToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {return tokenType + ":" + value;}
    }
    record PreprocessorToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {return tokenType + ":" + value;}
    }
    record AssignToken(TOKEN_TYPE tokenType, String value) implements Token {
        public String toString() {return tokenType + ":" + value;}
    }

    record ArithmeticOperator(TOKEN_TYPE tokenType, String value) implements Token {
       // https://www.codingeek.com/tutorials/c-programming/precedence-and-associativity-of-operators-in-c/
        enum Precedence{
            Multiplicative, Additive
        }
        public Precedence precedence(){
            return switch (value){
                case "*","/"->Precedence.Multiplicative; // multiplacative
                case "-","+"->Precedence.Additive; // additive
                default -> throw new IllegalStateException("Unexpected value: " + value);
            };
        }
        public String toString() {return tokenType + ":" + value + " "+precedence();}
    }


    enum TOKEN_TYPE {
        NONE(null),
        WHITE_SPACE("([ \n\t]+|//[^\n]*\n)"),
        RESERVED("(in|out|mainImage|return)(?![a-zA-Z0-9_])"),
        UNIFORM("(iTime|iResolution|iMouse)"),
        CREATE("(ivec[234]|vec[234]|imat[234]|mat[234])(?=[({])"),
        MATH_LIB_CALL("(abs|clamp|cos|dot|exp|length|normalize|normal|max|min|mix|pow|reflect|sin|sqrt|tan)(?=[({])"),
        CALL("([a-zA-Z_][a-zA-Z0-9_]*)(?=[({])"),
        PRIMITIVE_TYPE("(float|int|void)(?![a-zA-Z0-9_])"),
        VEC_TYPE("(ivec[234]|vec[234])(?![a-zA-Z0-9_])"),
        MAT_TYPE("(imat[234]|mat[234])(?![a-zA-Z0-9_])"),
        PRE_PREPROCESSOR("(#define|#include)"),

        IDENTIFIER("([a-zA-Z_][a-zA-Z0-9_]*)"),
        I32_OR_F32_CONST("(\\d+\\.?\\d*)"),
        OSYMBOL("([{(])"),
        CSYMBOL("([})])"),

        ASSIGN("(=|\\+=|\\-=|\\*=|/=)"),
        ARITHMETIC_OPERATOR("([+/\\-*])"),
        SEMICOLON("(;)"),
        COMMA("(,)"),
        SYMBOL("([.])");

        String regex;

        TOKEN_TYPE(String regex) {
            this.regex = regex;
        }

        static Token getToken(Matcher matcher) {
           // String weMatched = matcher.group();
           // System.out.println("We matched "+weMatched.replace("\n","\\n").replace("\t", "\\t"));
            for (var tokenType:TOKEN_TYPE.values()) {
                if (!tokenType.equals(NONE)) {
                    String val = matcher.group(tokenType.ordinal());
                    if (val != null && !val.isEmpty()) {
                        var token =  switch (tokenType) {
                            case WHITE_SPACE ->new WhiteSpaceToken(tokenType,val);
                            case MAT_TYPE -> new MatTypeToken(tokenType, val);
                            case VEC_TYPE -> new VecTypeToken(tokenType, val);
                            case PRIMITIVE_TYPE -> new PrimitiveTypeToken(tokenType, val);
                            case PRE_PREPROCESSOR -> new PreprocessorToken(tokenType, val);
                            case MATH_LIB_CALL -> new MathLibCallToken(tokenType, val);
                            case CALL -> new CallToken(tokenType, val);
                            case CREATE -> new CreateToken(tokenType, val);
                            case UNIFORM -> new UniformToken(tokenType, val);
                            case RESERVED -> new ReservedToken(tokenType, val);
                            case IDENTIFIER -> new SimpleToken(tokenType, val);
                            case I32_OR_F32_CONST -> val.contains(".") ? new FloatToken(tokenType,val,Float.parseFloat(val)) :new IntToken(tokenType,val,Integer.parseInt(val));
                            case OSYMBOL -> new SimpleToken(tokenType, val);
                            case CSYMBOL -> new SimpleToken(tokenType, val);
                            case ARITHMETIC_OPERATOR -> new ArithmeticOperator(tokenType, val);
                            case SEMICOLON -> new SimpleToken(tokenType, val);
                            case COMMA -> new SimpleToken(tokenType, val);
                            case ASSIGN -> new AssignToken(tokenType, val);
                            case SYMBOL -> new SimpleToken(tokenType, val);
                            case NONE -> new SimpleToken(tokenType, val);
                        };
                       // System.out.println(token);
                        return token;
                    }
                }
            }
            throw new RuntimeException("No token ");
        }
        static  Regex createPattern() {
            StringBuilder regexBuilder = new StringBuilder();
            for (var token:TOKEN_TYPE.values()){
                if (!token.equals(NONE)){
                    if (!regexBuilder.isEmpty()){
                        regexBuilder.append("|");
                    }
                    regexBuilder.append(token.regex);
                }
            }
            String regex = regexBuilder.toString();
            return Regex.of(regex);
        }
    }

    private static List<Token> tokenize(String source) {
        List<Token> tokens = new ArrayList<>();
        Regex TOKEN_PATTERN = TOKEN_TYPE.createPattern();
        Matcher matcher = TOKEN_PATTERN.pattern().matcher(source);
        while (matcher.find()) {
           // System.out.println("Looking at '"+matcher.group().replace("\n","\\n").replace("\t", "\\t")+"'");
            tokens.add(TOKEN_TYPE.getToken(matcher));
        }
        return tokens;
    }

    // Example parse: print declarations and functions
    private static void parse(List<Token> tokens) {
        int i = 0;
        while (i < tokens.size()) {
            Token t = tokens.get(i);
            // Very basic: look for [type] [identifier] ...
            if (t instanceof TypeToken typeToken  && i + 2 < tokens.size()) {
                Token ident = tokens.get(i + 1);
                Token next = tokens.get(i + 2);
                if (next.value().equals("(")) {
                    // Parse function definition head
                    System.out.println("Function found: returnType=" + t + ", name=" + ident);
                } else if (next.value().equals(";") || next.value().equals("=")) {
                    // Parse variable declaration
                    System.out.println("Variable found: type=" + t + ", name=" + ident);
                }
            }
            i++;
        }
    }


    // Example usage
    public static void main(String[] args) {
        String glslSource = """

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
        List<Token> tokens = tokenize(glslSource);
        var ansi = ANSI.of(System.out);

        tokens.forEach(token -> {
                    switch (token) {
                        case WhiteSpaceToken _ -> ansi.color(ANSI.GREEN, a -> a.apply(token.value()));
                        case ConstToken _  -> ansi.color(ANSI.YELLOW, a -> a.apply(token.value()));
                        case TypeToken _  -> ansi.color(ANSI.BLUE, a -> a.apply(token.value()));
                        case ReservedToken _  -> ansi.color(ANSI.CYAN, a -> a.apply(token.value()));
                        case PreprocessorToken _  -> ansi.color(ANSI.CYAN, a -> a.apply(token.value()));
                        case ArithmeticOperator _  -> ansi.color(ANSI.RED, a -> a.apply(token.value()));
                        case AssignToken _  -> ansi.color(ANSI.RED, a -> a.apply(token.value()));
                        case UniformToken _  -> ansi.color(ANSI.CYAN, a -> a.apply(token.value()));
                        case CallToken _  -> ansi.color(ANSI.PURPLE, a -> a.apply(token.value()));
                        case MathLibCallToken _  -> ansi.color(ANSI.BLUE, a -> a.apply(token.value()));
                        default -> ansi.apply(token.value());
                    }
                });
        parse(tokens);
    }

}
