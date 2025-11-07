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
import java.util.List;
import java.util.regex.Matcher;

class EliteMeshReader {

    interface State {

        String hexRegexStr = "((?:-?&[0-9a-fA-F][0-9a-fA-F])|0)";
        String commaRegexStr = " *, *";
        String hexOrColorCommaRegexStr = "(" + hexRegexStr + "|(?:(?:[a-zA-Z][a-zA-Z0-9]*)))" + commaRegexStr;
        String hexCommaRegexStr = hexRegexStr + commaRegexStr;
        String decRegexStr = "([0-9]+)";
        String decCommaRegexStr = decRegexStr + commaRegexStr;

        Regex face6Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "6", commaRegexStr, decCommaRegexStr.repeat(5), decRegexStr, " *$");
        Regex face5Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "5", commaRegexStr, decCommaRegexStr.repeat(4), decRegexStr, " *$");
        Regex face4Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "4", commaRegexStr, decCommaRegexStr.repeat(3), decRegexStr, " *$");
        Regex face3Regex = Regex.of("^ *", hexOrColorCommaRegexStr, hexCommaRegexStr.repeat(3), "3", commaRegexStr, decCommaRegexStr.repeat(2), decRegexStr, " *$");

        Regex vertexRegex = Regex.of("^ *" + hexCommaRegexStr + hexCommaRegexStr + hexRegexStr + " *$");
        Regex emptyRegex = Regex.of("^ *$");
        Regex remRegex = Regex.of("^ *REM(.*)$");
        Regex colonRegex = Regex.of("^ *(:) *$");
        Regex facesRegex = Regex.of("^ *(faces) *$");

        default String name() {
            return this.getClass().getSimpleName();
        }

        record awaiting_name(Regex r) implements State {
        }

        record awaiting_lazer(Regex r) implements State {
        }

        record awaiting_counts(Regex r) implements State {
        }

        record awaiting_vertices(Regex r) implements State {
        }

        record awaiting_hue_lig_sat(Regex r) implements State {
        }

        record awaiting_faces() implements State {
            public Regex r() {
                return null;
            }
        }

        record done() implements State {
            public Regex r() {
                return null;
            }
        }

        record none() implements State {
            public Regex r() {
                return null;
            }
        }

        awaiting_name awaiting_name = new awaiting_name(Regex.of("^ *([A-Za-z][0-9A-Za-z]+) *$"));
        awaiting_lazer awaiting_lazer = new awaiting_lazer(Regex.of("^ *" + hexRegexStr + " *$"));
        awaiting_counts awaiting_counts = new awaiting_counts(Regex.of("^ *" + hexCommaRegexStr + hexRegexStr + " *$"));
        awaiting_vertices awaiting_vertices = new awaiting_vertices(Regex.of("^ *(vertices) *$"));
        awaiting_hue_lig_sat awaiting_hue_lig_sat = new awaiting_hue_lig_sat(Regex.of("^ *(hue-lig-sat) *$"));
        awaiting_faces awaiting_faces = new awaiting_faces();
        done done = new done();
        none none = new none();

        class Machine {
            State state = none;

            private Machine state(State state) {
                //  System.out.println("from " + this.state.name() + " to " + state.name());
                this.state = state;
                return this;
            }

            Machine() {
                state(none);
            }

            Machine awaiting_name() {
                return state(awaiting_name);
            }

            Machine awaiting_lazer() {
                return state(awaiting_lazer);
            }

            Machine awaiting_counts() {
                return state(awaiting_counts);
            }

            Machine awaiting_vertices() {
                return state(awaiting_vertices);
            }

            Machine awaiting_faces() {
                return state(awaiting_faces);
            }

            Machine awaiting_hue_lig_sat() {
                return state(awaiting_hue_lig_sat);
            }

            Machine done() {
                return state(done);
            }
        }
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

    record S32xN (Regex regex, Matcher matcher, boolean matched) implements Regex.OK {
        S32xN(Regex r, Matcher m) {
            this(r, m, true);
        }
        int count(){
            return matcher.groupCount();
        }
    }
/*
    record hex(F32.Vec3 v0,F32.Vec3 v1,F32.Vec3 v2,F32.Vec3 v3,F32.Vec3 v4,F32.Vec3 v5){
        static hex of(List<F32.Vec3> vecEntries, S32xN s32xN){
            int i6 = s32xN.asInt(6);
            int i7 = s32xN.asInt(7);
            int i8 = s32xN.asInt(8);
            int i9 = s32xN.asInt(9);
            int i10 = s32xN.asInt(10);
            int i11 = s32xN.asInt(11);
            return new hex(
                    vecEntries.get(i6),
                    vecEntries.get(i7),
                    vecEntries.get(i8),
                    vecEntries.get(i9),
                    vecEntries.get(i10),
                    vecEntries.get(i11)
            );

        }
    }
    record pent(F32.Vec3 v0,F32.Vec3 v1,F32.Vec3 v2,F32.Vec3 v3,F32.Vec3 v4){
        static pent of(List<F32.Vec3> vecEntries, S32xN s32xN){
            int i6 = s32xN.asInt(6);
            int i7 = s32xN.asInt(7);
            int i8 = s32xN.asInt(8);
            int i9 = s32xN.asInt(9);
            int i10 = s32xN.asInt(10);
            return new pent(
                    vecEntries.get(i6),
                    vecEntries.get(i7),
                    vecEntries.get(i8),
                    vecEntries.get(i9),
                    vecEntries.get(i10)
            );

        }
    }
    record quad(F32.Vec3 v0,F32.Vec3 v1,F32.Vec3 v2,F32.Vec3 v3){
        static quad of(List<F32.Vec3> vecEntries, S32xN s32xN){
            int i6 = s32xN.asInt(6);
            int i7 = s32xN.asInt(7);
            int i8 = s32xN.asInt(8);
            int i9 = s32xN.asInt(9);
            return new quad(
                    vecEntries.get(i6),
                    vecEntries.get(i7),
                    vecEntries.get(i8),
                    vecEntries.get(i9)
            );

        }
    }
    record tri(F32.Vec3 v0,F32.Vec3 v1,F32.Vec3 v2){
        static tri of(List<F32.Vec3> vecEntries, S32xN s32xN){
            int i6 = s32xN.asInt(6);
            int i7 = s32xN.asInt(7);
            int i8 = s32xN.asInt(8);
            return new tri(
                    vecEntries.get(i6),
                    vecEntries.get(i7),
                    vecEntries.get(i8)
            );

        }
    }
*/
    void load(String name) {
        final var oldMesh = StreamMutable.of((F32Mesh3D) null);
      //  final var newMesh = StreamMutable.of((F32.Mesh) null);
        final var sm = new State.Machine().awaiting_name();
        new BufferedReader(
                new InputStreamReader(EliteMeshReader.class.getResourceAsStream("/meshes/Elite.txt"), StandardCharsets.UTF_8))
                .lines()
                .map(String::trim)
                .forEach(line -> {
                    switch(sm.state){
                        case State.awaiting_name s when s.r().matches(line, whoseMatcher -> whoseMatcher.group(1).equals(name))->{
                            sm.awaiting_lazer();
                                oldMesh.set(F32Mesh3D.of(name));
                        }
                        case State.awaiting_lazer s when s.r().matches(line) -> sm.awaiting_counts();
                        case State.awaiting_counts s when s.r().matches(line) -> sm.awaiting_vertices();
                        case State.awaiting_vertices s when s.r().matches(line) -> sm.awaiting_faces();
                        case State.awaiting_faces _ when State.vertexRegex.is(line, F32x3::new) instanceof F32x3 f32x3 ->{

                                oldMesh.get().vec3(f32x3.f(1), f32x3.f(2), f32x3.f(3));

                        }
                        case State.awaiting_faces _ when State.facesRegex.matchesOrThrow(line) ->
                                sm.awaiting_hue_lig_sat();
                        case State.awaiting_hue_lig_sat _ when State.face6Regex.is(line, S32xN::new) instanceof S32xN s32xN ->{

                                oldMesh.get().hex(
                                        oldMesh.get().vecEntries.get(s32xN.asInt(6)).idx(),
                                        oldMesh.get().vecEntries.get(s32xN.asInt(7)).idx(),
                                        oldMesh.get().vecEntries.get(s32xN.asInt(8)).idx(),
                                        oldMesh.get().vecEntries.get(s32xN.asInt(9)).idx(),
                                        oldMesh.get().vecEntries.get(s32xN.asInt(10)).idx(),
                                        oldMesh.get().vecEntries.get(s32xN.asInt(11)).idx(),
                                        0xff7f00);

                        }
                        case State.awaiting_hue_lig_sat _ when State.face5Regex.is(line, S32xN::new) instanceof S32xN s32xN ->{

                                   oldMesh.get().pent(
                                           oldMesh.get().vecEntries.get(s32xN.asInt(6)).idx(),
                                           oldMesh.get().vecEntries.get(s32xN.asInt(7)).idx(),
                                           oldMesh.get().vecEntries.get(s32xN.asInt(8)).idx(),
                                           oldMesh.get().vecEntries.get(s32xN.asInt(9)).idx(),
                                           oldMesh.get().vecEntries.get(s32xN.asInt(10)).idx(),
                                           0x7fff00);

                        }
                        case State.awaiting_hue_lig_sat _ when State.face4Regex.is(line, S32xN::new) instanceof S32xN s32xN ->{

                                oldMesh.get().quad(
                                        oldMesh.get().vecEntries.get(s32xN.asInt(6)).idx(),
                                        oldMesh.get().vecEntries.get(s32xN.asInt(7)).idx(),
                                        oldMesh.get().vecEntries.get(s32xN.asInt(8)).idx(),
                                        oldMesh.get().vecEntries.get(s32xN.asInt(9)).idx(),
                                        0x00ff7f);

                        }
                        case State.awaiting_hue_lig_sat _ when State.face3Regex.is(line, S32xN::new) instanceof S32xN s32xN ->{

                                oldMesh.get().tri(
                                        oldMesh.get().vecEntries.get(s32xN.asInt(6)).idx(),
                                        oldMesh.get().vecEntries.get(s32xN.asInt(7)).idx(),
                                        oldMesh.get().vecEntries.get(s32xN.asInt(8)).idx(),
                                        0x007fff);

                        }
                        case State.awaiting_hue_lig_sat s when s.r().matches(line) -> {
                                oldMesh.get().fin();
                            sm.done();
                        }
                        case State.awaiting_hue_lig_sat _ when !State.remRegex.matches(line) ->
                                System.out.println("UNHANDLED " + line);

                        case State.done _-> {}
                        case State _ when Regex.any(line, State.remRegex, State.emptyRegex, State.colonRegex).matched()->{}
                        case State _ ->{}
                    }
                });
    }

}
