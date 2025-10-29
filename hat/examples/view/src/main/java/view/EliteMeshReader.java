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

import hat.util.Regex;
import hat.util.StreamMutable;
import view.f32.F32Mesh3D;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;

class EliteMeshReader {
    static String hexRegexStr = "((?:-?&[0-9a-fA-F][0-9a-fA-F])|0)";
    static String commaRegexStr = " *, *";
    static String hexOrColorCommaRegexStr = "(" + hexRegexStr + "|(?:(?:[a-zA-Z][a-zA-Z0-9]*)))" + commaRegexStr;
    static String hexCommaRegexStr = hexRegexStr + commaRegexStr;
    static String decRegexStr = "([0-9]+)";
    static String decCommaRegexStr = decRegexStr + commaRegexStr;

    static Regex face6Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "6", commaRegexStr, decCommaRegexStr.repeat(5), decRegexStr, " *$");
    static Regex face5Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "5", commaRegexStr, decCommaRegexStr.repeat(4), decRegexStr, " *$");
    static Regex face4Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "4", commaRegexStr, decCommaRegexStr.repeat(3), decRegexStr, " *$");
    static Regex face3Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "3", commaRegexStr, decCommaRegexStr.repeat(2), decRegexStr, " *$");

    static Regex vertexRegex = Regex.of("^ *" + hexCommaRegexStr + hexCommaRegexStr + hexRegexStr + " *$");
    static Regex emptyRegex = Regex.of("^ *$");
    static Regex remRegex = Regex.of("^ *REM(.*)$");
    static Regex colonRegex = Regex.of("^ *(:) *$");
    static Regex facesRegex = Regex.of("^ *(faces) *$");

    interface State {
        default String name(){
            return this.getClass().getSimpleName();
        }
        record awaiting_name(Regex r) implements State {}
        record awaiting_lazer(Regex r) implements State {}
        record awaiting_counts(Regex r) implements State {}
        record awaiting_vertices(Regex r) implements State {}
        record awaiting_hue_lig_sat(Regex r) implements State {}
        record awaiting_faces() implements State { }
        record done() implements State {}
        awaiting_name awaiting_name = new awaiting_name( Regex.of("^ *([A-Za-z][0-9A-Za-z]+) *$"));
        awaiting_lazer awaiting_lazer = new awaiting_lazer(Regex.of("^ *" + hexRegexStr + " *$"));
        awaiting_counts awaiting_counts = new awaiting_counts(Regex.of("^ *" + hexCommaRegexStr + hexRegexStr + " *$"));
        awaiting_vertices awaiting_vertices = new awaiting_vertices(Regex.of("^ *(vertices) *$"));
        awaiting_hue_lig_sat awaiting_hue_lig_sat = new awaiting_hue_lig_sat(Regex.of("^ *(hue-lig-sat) *$"));
        awaiting_faces awaiting_faces = new awaiting_faces();
        done done  = new done();
        //St[] all = new St[]{awaiting_name, awaiting_lazer, awaiting_counts, awaiting_vertices, awaiting_hue_lig_sat, awaiting_faces};
    }
    record F32x3(Regex regex, Matcher matcher, boolean matched) implements Regex.OK {
        float f(int idx) {
            var s = string(idx);
            return (s.startsWith("-")) ?
                    (-Integer.parseInt(s.substring(2), 16) / 64f) : (Integer.parseInt(s.substring(1), 16) / 64f);
        }

        F32x3(Regex r, Matcher m) {
            this(r, m, true);
        }
    }

    static void load(String name) {
        final var mesh = StreamMutable.of((F32Mesh3D) null);
        final var st = StreamMutable.of((State) State.awaiting_name);
        new BufferedReader(
                new InputStreamReader(EliteMeshReader.class.getResourceAsStream("/meshes/Elite.txt"), StandardCharsets.UTF_8))
                .lines()
                .map(String::trim)
                .forEach(line -> {
                    switch(st.get()){
                        case State.awaiting_name s when s.r().matches(line, whoseMatcher -> whoseMatcher.group(1).equals(name))->{
                            st.set(State.awaiting_lazer);
                            mesh.set(F32Mesh3D.of(name));
                        }
                        case State.awaiting_lazer s-> st.setIf(s.r.matches(line), State.awaiting_counts);
                        case State.awaiting_counts s-> st.setIf(s.r.matches(line), State.awaiting_vertices);
                        case State.awaiting_vertices s-> st.setIf(s.r.matches(line), State.awaiting_faces);
                        case State.awaiting_faces _-> {
                            if (vertexRegex.is(line,F32x3::new) instanceof F32x3 ok) {
                                mesh.get().vec3(ok.f(1), ok.f(2), ok.f(3));
                            } else if (facesRegex.matchesOrThrow(line)) {
                                st.set(State.awaiting_hue_lig_sat);
                            }
                        }
                        case State.awaiting_hue_lig_sat s-> {
                            if (Regex.any(line, face6Regex, face5Regex, face4Regex, face3Regex) instanceof Regex.OK ok) {
                                int v0 = mesh.get().vecEntries[ok.asInt(6)];
                                int v1 = mesh.get().vecEntries[ok.asInt(7)];
                                int v2 = mesh.get().vecEntries[ok.asInt(8)];
                                if (ok.regex() == face3Regex) {
                                    mesh.get().tri(v0, v1, v2, 0x00ff00);
                                } else if (ok.regex() == face4Regex) {
                                    mesh.get().quad(v0, v1, v2, mesh.get().vecEntries[ok.asInt(9)], 0xff0000);
                                } else if (ok.regex() == face5Regex) {
                                    mesh.get().pent(v0, v1, v2, mesh.get().vecEntries[ok.asInt(9)], mesh.get().vecEntries[ok.asInt(10)], 0x0000ff);
                                } else {
                                    mesh.get().hex(v0, v1, v2, mesh.get().vecEntries[ok.asInt(9)], mesh.get().vecEntries[ok.asInt(10)], mesh.get().vecEntries[ok.asInt(11)], 0xfff000);
                                }
                            } else if (s.r().matches(line)) {
                                mesh.get().fin();
                                st.set(State.done);
                            }else if (!remRegex.matches(line)){
                                System.out.println("UNHANDLED "+line);
                            }
                        }
                        case State.done _-> {}
                        case State _ when Regex.any(line, remRegex, emptyRegex, colonRegex).matched()->{}
                        case State _ ->{}
                    }
                });
    }
}
