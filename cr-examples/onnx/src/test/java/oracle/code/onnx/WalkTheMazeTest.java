/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

package oracle.code.onnx;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.util.List;
import jdk.incubator.code.CodeReflection;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static java.util.Optional.empty;
import static oracle.code.onnx.OnnxOperators.*;
import static oracle.code.onnx.OnnxRuntime.execute;

public class WalkTheMazeTest {

    final String expectedPath;

    // initializers
    final Tensor<Byte> maze;
    final Tensor<Boolean> _true;
    final Tensor<Long> homePos, directionNorth, directionSouth, directionEast, directionWest, oneOne, three, limit,
                       stepSouth, stepNorth, stepEast, stepWest, scalarShape, wall;

    public WalkTheMazeTest() {
        expectedPath = ">>^^>>vv>>>>>><<^^^^^<<<<<<^^>>>>>>>>>>vv<<vvv>>vv>>>>^^>>^<<^^>><<^^>><<<<^^>>>>^^>>vvvvvvvvv>>vv<<<<>>>>^^^<<^^>><<^^"
                     + ">>>>vv>>>>>><<^^>>>><<<<vv<<^^^^>>>>>>>>vvvv<<vv<<<<<<<<vvv>>>>>><<<<<<^^>>>>>>>>vv>>>>>>>>^^^<<<<^^>>>>>>>>^^^^^^<vv<<"
                     + "<<<<<<<^^>>>>>>><<<^^<<^^^^>>>>vv<<>>vv>>>>^^<<<<^^>>>><<^^>><<<<vv<<<<vv<<<<<<^^>>>>^^<<<<<<vvvv<<^^<<<<^^>>>><<<<vvvv"
                     + "<<^^<<<<^^>>>><<<<vvvvvv<<^^vvvv>>>>vv<<<<vvvv^^>>vvvv<<v>>vv<<<<^^^<<^^>>^^<<<<^^^^^^<<<<>>vv<<<<^^<<^^>>^^^^>>vv>>>>>"
                     + ">^^<<<<>>>>vv<<<<<<^^<<<<vvvvvv>>vv<<vv>>>>>>>>vv<<<<<<vv>>>>vvvvv<<^^^<<<<^^^^vvvvv>>vv<<>";

        var arena = Arena.ofAuto();

        maze = Tensor.ofShape(arena, new long[]{22, 48},
                """
                ###############################################
                #     #     #   #     #     #       #   #     #
                # # # ##### # ### ##### ##### ##### ##### # ###
                # # #       #   #     #     # #     #         #
                # # ######### ### ### # ### # # ##### ### #####
                #   #         # # #       #   #       #       #
                # ########### # # # # ####### ### ### ### ##  #
                #   #     #   #     #       #     #     #     #
                ### ### # # ### ############### # ##### #######
                #       # #   #       #         # #        #  #
                # ####### ### ##### # # ##### ##### ########  #
                #         #   #     # # # #                   #
                ######### ##### ##### ### # ####### ########  #
                # #           #     #     # #     #   #       #
                # # ######### # # ### ### # # ##### # ######  #
                # #       #   # #   #   #       #   #         #
                # ##### # # ##### ### ########### ### #########
                #     # # #   #     #   #           #     #   #
                #     # # #   #     #   #           #     #   #
                ### # # # ### ### ##### # ####### ### ### #   #
                #   #       #     #     #       #         #   #
                ###############################################
                """.getBytes());

        _true = Tensor.ofScalar(arena, true);
        homePos = Tensor.ofFlat(arena, 20l, 1); // bottom left corner
        directionNorth = Tensor.ofFlat(arena, '^');
        directionSouth = Tensor.ofFlat(arena, 'v');
        directionEast = Tensor.ofFlat(arena, '>');
        directionWest = Tensor.ofFlat(arena, '<');
        oneOne = Tensor.ofFlat(arena, 1l, 1);
        three = Tensor.ofFlat(arena, 3l);
        limit = Tensor.ofFlat(arena, 1000l);
        stepSouth = Tensor.ofFlat(arena, 1l, 0);
        stepNorth = Tensor.ofFlat(arena, -1l, 0);
        stepEast = Tensor.ofFlat(arena, 0l, 1);
        stepWest = Tensor.ofFlat(arena, 0l, -1);
        scalarShape = Tensor.ofFlat(arena, new long[0]);
        wall = Tensor.ofScalar(arena, '#');
    }

    @CodeReflection
    public Tensor<Long> turnLeft(Tensor<Long> direction) {
        return If(Equal(direction, directionEast),
                () -> Identity(directionNorth),
                () -> If(Equal(direction, directionNorth),
                        () -> Identity(directionWest),
                        () -> If(Equal(direction, directionWest),
                            () -> Identity(directionSouth),
                            () -> Identity(directionEast))));
    }

    @CodeReflection
    public Tensor<Long> turnRight(Tensor<Long> direction) {
        return Loop(three, _true, direction, (i, cond, d)
                -> new LoopResult<>(cond, turnLeft(d)));
    }

    @CodeReflection
    public Tensor<Boolean> isWallAt(Tensor<Long> pos) {
        return Equal(CastLike(Slice(maze, pos, Add(pos, oneOne), empty(), empty()), wall, empty(), empty()), wall);
    }

    @CodeReflection
    public Tensor<Long> step(Tensor<Long> myPos, Tensor<Long> myDirection) {
        return  If(Equal(myDirection, directionEast),
                () -> Add(myPos, stepEast),
                () -> If(Equal(myDirection, directionNorth),
                        () -> Add(myPos, stepNorth),
                        () -> If(Equal(myDirection, directionWest),
                            () -> Add(myPos, stepWest),
                            () -> Add(myPos, stepSouth))));
    }

    @CodeReflection
    public Tensor<Boolean> atHome(Tensor<Long> pos) {
        return ReduceMin(Equal(pos, homePos), empty(), empty(), empty());
    }

    @CodeReflection
    public Tensor<Long> turnLeftWhileWall(Tensor<Long> pos, Tensor<Long> direction) {
        var initialCond = Reshape(isWallAt(step(pos, direction)), scalarShape, empty());
        return Loop(limit, initialCond, direction, (_, _, dir) -> {
                dir = turnLeft(dir);
                return new LoopResult<>(isWallAt(step(pos, dir)), dir);
            });
    }

    @CodeReflection
    public Tensor<Byte> appendToPath(Tensor<Byte> path, Tensor<Long> direction) {
        return Concat(List.of(path, Cast(direction, empty(), 2, empty())), 0);
    }

    public record LoopData(Tensor<Long> pos, Tensor<Long> direction, Tensor<Byte> path) {}

    @CodeReflection
    public Tensor<Byte> walkAroundTheMaze() {
        var initData = new LoopData(homePos, directionEast, Cast(directionEast, empty(), 2, empty()));
        var outData = Loop(limit, _true, initData, (_, _, loopData) -> {
            // walk along the right wall
            var pos = step(loopData.pos(), loopData.direction());
            var direction = turnRight(loopData.direction());
            direction = turnLeftWhileWall(pos, direction);

            return new LoopResult<>(Not(atHome(pos)), new LoopData(pos, direction, appendToPath(loopData.path(), direction)));
        });
        return outData.path();
    }

    @Test
    public void testWalkAroundTheMaze() throws Exception {
        try (var arena = Arena.ofConfined()) {
            var directions = execute(arena, MethodHandles.lookup(), () -> walkAroundTheMaze());
            Assertions.assertEquals(expectedPath, new String(directions.data().toArray(ValueLayout.JAVA_BYTE)));
        }
    }
}
