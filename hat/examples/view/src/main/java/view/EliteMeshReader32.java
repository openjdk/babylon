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

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

class EliteMeshReader32 implements EliteReader{
    @Override
    public void load(String name) {
        final var mesh = StreamMutable.of((F32.Mesh) null);
        final var sm = new State.Machine().awaiting_name();
        new BufferedReader(
                new InputStreamReader(EliteMeshReader.class.getResourceAsStream("/meshes/Elite.txt"), StandardCharsets.UTF_8))
                .lines()
                .map(String::trim)
                .forEach(line -> {
                    switch (sm.state) {
                        case State.awaiting_name s when s.r().matches(line, whoseMatcher ->
                                whoseMatcher.group(1).equals(name)) -> {
                            sm.awaiting_lazer();
                            mesh.set(F32.Mesh.of(name));
                        }
                        case State.awaiting_lazer s when s.r().matches(line) -> sm.awaiting_counts();
                        case State.awaiting_counts s when s.r().matches(line) -> sm.awaiting_vertices();
                        case State.awaiting_vertices s when s.r().matches(line) -> sm.awaiting_faces();
                        case State.awaiting_faces _ when State.vertexRegex.is(line, F32x3::new) instanceof F32x3 f32x3 ->
                                mesh.get().vec3(f32x3.f(1), f32x3.f(2), f32x3.f(3));
                        case State.awaiting_faces _ when State.facesRegex.matchesOrThrow(line) ->
                                sm.awaiting_hue_lig_sat();
                        case State.awaiting_hue_lig_sat _ when State.face6Regex.is(line, S32xN::new) instanceof S32xN s32xN ->{
                                var h = hex.of(mesh.get().vecEntries,s32xN);
                                mesh.get().hex(h.v0(),h.v1(),h.v2(),h.v3(),h.v4(),h.v5(), 0xfff000);
                        }
                        case State.awaiting_hue_lig_sat _ when State.face5Regex.is(line, S32xN::new) instanceof S32xN s32xN -> {
                            var h = hex.of(mesh.get().vecEntries,s32xN);
                            mesh.get().pent(h.v0(),h.v1(),h.v2(),h.v3(),h.v4(), 0xfff000);
                        }
                        case State.awaiting_hue_lig_sat _ when State.face4Regex.is(line,S32xN::new) instanceof S32xN s32xN ->{
                            var h = quad.of(mesh.get().vecEntries,s32xN);
                            mesh.get().quad(h.v0(),h.v1(),h.v2(),h.v3(), 0xfff000);
                        }
                        case State.awaiting_hue_lig_sat _ when State.face3Regex.is(line, S32xN::new) instanceof S32xN s32xN -> {
                            var h = tri.of(mesh.get().vecEntries,s32xN);
                            mesh.get().tri(h.v0(),h.v1(),h.v2(), 0xfff000);
                        }
                        case State.awaiting_hue_lig_sat s when s.r().matches(line) -> {
                            mesh.get().fin();
                            sm.done();
                        }
                        case State.awaiting_hue_lig_sat _ when !State.remRegex.matches(line) ->
                                System.out.println("UNHANDLED " + line);
                        case State.done _ -> {
                        }
                        case State _ when Regex.any(line, State.remRegex, State.emptyRegex, State.colonRegex).matched() -> {
                        }
                        case State _ -> {/*no state change*/ }
                    }
                });
    }
}

