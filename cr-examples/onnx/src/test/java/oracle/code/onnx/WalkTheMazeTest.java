package oracle.code.onnx;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.Optional;
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
    final Tensor<Long> homePos, directionNorth, directionSouth, directionEast, directionWest,
                       oneOne, zero, two, three, mOne, mThree, max, limit,
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
        zero = Tensor.ofFlat(arena, 0l);
        two = Tensor.ofFlat(arena, 2l);
        three = Tensor.ofFlat(arena, 3l);
        mOne = Tensor.ofFlat(arena, -1l);
        mThree = Tensor.ofFlat(arena, -3l);
        max = Tensor.ofFlat(arena, Long.MAX_VALUE);
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
                -> LoopReturn(cond, turnLeft(d)));
    }

    @CodeReflection
    public Tensor<Boolean> isWallAt(Tensor<Long> pos) {
        return Equal(CastLike(Slice(maze, pos, Add(pos, oneOne), empty(), empty()), wall, empty()), wall);
    }

    @CodeReflection
    public Tensor<Long> posInFrontOfMe(Tensor<Long> myPos, Tensor<Long> myDirection) {
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
    public Tensor<Long> lastPos(Tensor<Long> pathLog) {
        return Slice(pathLog, mThree, mOne, empty(), empty());
    }

    @CodeReflection
    public Tensor<Long> lastDirection(Tensor<Long> pathLog) {
        return Slice(pathLog, mOne, max, empty(), empty());
    }

    @CodeReflection
    public Tensor<Long> addToLog(Tensor<Long> pathLog, Tensor<Long> pos, Tensor<Long> direction) {
        return Concat(List.of(pathLog, pos, direction), 0);
    }

    @CodeReflection
    public Tensor<Byte> extractDirections(Tensor<Long> pathLog) {
        return Cast(Slice(pathLog, two, max, Optional.of(zero), Optional.of(three)), empty(), 3);
    }

    @CodeReflection
    public Tensor<Long> turnLeftWhileWall(Tensor<Long> pos, Tensor<Long> direction) {
        var initialCond = Reshape(isWallAt(posInFrontOfMe(pos, direction)), scalarShape, empty());
        return Loop(limit, initialCond, direction, (_, _, dir) -> {
                dir = turnLeft(dir);
                return LoopReturn(isWallAt(posInFrontOfMe(pos, dir)), dir);
            });
    }

    @CodeReflection
    public Tensor<Long> walkAroundTheMaze() {
        var start = Concat(List.of(homePos, directionEast), 0);
        var pathLog = Loop(limit, _true, start, (_, _, log) -> {
            var pos = lastPos(log);
            var direction = lastDirection(log);

            // walk along the right wall
            pos = posInFrontOfMe(pos, direction);
            direction = turnRight(direction);
            direction = turnLeftWhileWall(pos, direction);

            return LoopReturn(Not(atHome(pos)), addToLog(log, pos, direction));
        });
        return pathLog;
    }

    @Test
    public void testWalkAroundTheMaze() throws Exception {
        try (var arena = Arena.ofConfined()) {
            var directions = execute(arena, MethodHandles.lookup(), () -> extractDirections(walkAroundTheMaze()));
            Assertions.assertEquals(expectedPath, new String(directions.data().toArray(ValueLayout.JAVA_BYTE)));
        }
    }
}
