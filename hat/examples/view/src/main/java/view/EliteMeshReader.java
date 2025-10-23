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

/*
 *  Based on mesh descriptions found here
 *      https://6502disassembly.com/a2-elite/
 *      https://6502disassembly.com/a2-elite/meshes.html
 *
 */
package view;

import view.f32.F32Mesh3D;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;

class EliteMeshReader {
    static Regex remRegex = Regex.of("^ *REM(.*)$");
    static Regex colonRegex = Regex.of("^ *(:) *$");
    static Regex verticesRegex = Regex.of( "^ *(vertices) *$");
    static Regex facesRegex = Regex.of("^ *(faces) *$");
    static Regex hueLigSatRegex = Regex.of("^ *(hue-lig-sat) *$");

    static String hexRegexStr = "((?:-?&[0-9a-fA-F][0-9a-fA-F])|0)";
    static String commaRegexStr = " *, *";
    static String hexOrColorCommaRegexStr = "(" + hexRegexStr + "|(?:(?:[a-zA-Z][a-zA-Z0-9]*)))" + commaRegexStr;
    static String hexCommaRegexStr = hexRegexStr + commaRegexStr;
    static String decRegexStr = "([0-9]+)";
    static String decCommaRegexStr = decRegexStr + commaRegexStr;

    static Regex face6Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "6" ,commaRegexStr, decCommaRegexStr.repeat(5), decRegexStr, " *$");
    static Regex face5Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "5" ,commaRegexStr, decCommaRegexStr.repeat(4), decRegexStr, " *$");
    static Regex face4Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "4" ,commaRegexStr, decCommaRegexStr.repeat(3), decRegexStr, " *$");
    static Regex face3Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "3" ,commaRegexStr, decCommaRegexStr.repeat(2), decRegexStr, " *$");

    static Regex frontLaserVertexRegex = Regex.of( "^ *" + hexRegexStr + " *$");
    static Regex vertexRegex = Regex.of("^ *" + hexCommaRegexStr + hexCommaRegexStr + hexRegexStr + " *$");
    static Regex vertexCountRegex = Regex.of("^ *" + hexCommaRegexStr + hexRegexStr + " *$");
    static Regex nameRegex = Regex.of("^ *([A-Za-z][0-9A-Za-z]+) *$");
    static Regex emptyRegex = Regex.of("^ *$");



    enum State {AWAITING_NAME, AWAITING_LAZER, AWAITING_COUNTS, AWAITING_VERTICES, AWAITING_HUE_LIG_SAT, AWAITING_FACES}

    static void load(String name) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(EliteMeshReader.class.getResourceAsStream("/meshes/Elite.txt")));

            State state = State.AWAITING_NAME;
            F32Mesh3D mesh= null;
            for (String line = reader.readLine(); line != null; line = reader.readLine()) {
                line = line.trim();
                if (!Regex.any(line, remRegex,emptyRegex,colonRegex).matched()) {
                    switch (state) {
                        case AWAITING_NAME ->{
                            if (nameRegex.is(line) instanceof Regex.OK ok && ok.string(1).equals(name)) {
                                state = State.AWAITING_LAZER;
                                mesh = F32Mesh3D.of(name);
                            }
                        }
                        case AWAITING_LAZER-> state = frontLaserVertexRegex.is(line).matched()?State.AWAITING_COUNTS:state;
                        case AWAITING_COUNTS-> state  =vertexCountRegex.is(line).matched()?State.AWAITING_VERTICES:state;
                        case AWAITING_VERTICES-> state = verticesRegex.is(line).matched()?State.AWAITING_FACES:state;
                        case AWAITING_FACES->{
                            if (vertexRegex.is(line) instanceof Regex.OK ok) {
                                mesh.vec3(ok.f(1),  ok.f(2), ok.f(3));
                            } else if (facesRegex.is(line).matched()) {
                                state = State.AWAITING_HUE_LIG_SAT;
                            }
                        }
                        case AWAITING_HUE_LIG_SAT-> {
                            if (Regex.any(line,face6Regex,face5Regex,face4Regex,face3Regex) instanceof Regex.OK ok) {
                                int v0 = mesh.vecEntries[ok.i(6)];
                                int v1 = mesh.vecEntries[ok.i(7)];
                                int v2 = mesh.vecEntries[ok.i(8)];

                                if (ok.regex() == face3Regex){
                                    mesh.tri(v0, v1, v2,  0x00ff00 );
                                }else if (ok.regex() == face4Regex) {
                                    mesh.quad(v0, v1,v2, mesh.vecEntries[ok.i(9)],  0xff0000);
                                } else if (ok.regex()== face5Regex) {
                                    mesh.pent(v0, v1, v2, mesh.vecEntries[ok.i(9)], mesh.vecEntries[ok.i(10)], 0x0000ff);
                                } else {
                                    mesh.hex(v0, v1, v2, mesh.vecEntries[ok.i(9)], mesh.vecEntries[ok.i(10)], mesh.vecEntries[ok.i(11)], 0xfff000);
                                }
                            } else if ((hueLigSatRegex.is(line).matched())) {
                                mesh.fin();
                                return;
                            } else {
                                System.out.println("In " + state + " skipping " + line);
                            }
                        }
                        default->  throw new IllegalStateException(("WHAt " + line));

                    }
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
