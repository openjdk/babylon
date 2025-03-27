package oracle.code.onnx;

import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import jdk.incubator.code.CodeReflection;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static java.util.Optional.empty;
import static oracle.code.onnx.OnnxOperators.*;
import static oracle.code.onnx.OnnxRuntime.execute;

public class WalkTheMazeTest {

    static final Tensor<Byte> MAZE = Tensor.ofShape(new long[]{22, 48},
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

    static final Tensor<Long> HOME_POS = Tensor.ofFlat(20l, 1); // bottom left corner

    static final String EXPECTED_PATH =
            ">>^^>>vv>>>>>><<^^^^^<<<<<<^^>>>>>>>>>>vv<<vvv>>vv>>>>^^>>^<<^^>><<^^>><<<<^^>>>>^^>>vvvvvvvvv>>vv<<<<>>>>^^^<<^^>><<^^>>>>vv>>>>>><<^^>>>><<<<vv<"
           +"<^^^^>>>>>>>>vvvv<<vv<<<<<<<<vvv>>>>>><<<<<<^^>>>>>>>>vv>>>>>>>>^^^<<<<^^>>>>>>>>^^^^^^<vv<<<<<<<<<^^>>>>>>><<<^^<<^^^^>>>>vv<<>>vv>>>>^^<<<<^^>>>"
           +"><<^^>><<<<vv<<<<vv<<<<<<^^>>>>^^<<<<<<vvvv<<^^<<<<^^>>>><<<<vvvv<<^^<<<<^^>>>><<<<vvvvvv<<^^vvvv>>>>vv<<<<vvvv^^>>vvvv<<v>>vv<<<<^^^<<^^>>^^<<<<^"
           +"^^^^^<<<<>>vv<<<<^^<<^^>>^^^^>>vv>>>>>>^^<<<<>>>>vv<<<<<<^^<<<<vvvvvv>>vv<<vv>>>>>>>>vv<<<<<<vv>>>>vvvvv<<^^^<<<<^^^^vvvvv>>vv<<>";

    static final Tensor<Long> DIRECTION_NORTH = Tensor.ofFlat('^');
    static final Tensor<Long> DIRECTION_SOUTH = Tensor.ofFlat('v');
    static final Tensor<Long> DIRECTION_EAST = Tensor.ofFlat('>');
    static final Tensor<Long> DIRECTION_WEST = Tensor.ofFlat('<');
    static final Tensor<Boolean> TRUE = Tensor.ofScalar(true);
    static final Tensor<Long> ONE_ONE = Tensor.ofFlat(1l, 1);
    static final Tensor<Long> ZERO = Tensor.ofFlat(0l);
    static final Tensor<Long> TWO = Tensor.ofFlat(2l);
    static final Tensor<Long> THREE = Tensor.ofFlat(3l);
    static final Tensor<Long> MONE = Tensor.ofFlat(-1l);
    static final Tensor<Long> MTHREE = Tensor.ofFlat(-3l);
    static final Tensor<Long> MAX = Tensor.ofFlat(Long.MAX_VALUE);
    static final Tensor<Long> LIMIT = Tensor.ofFlat(1000l);
    static final Tensor<Long> STEP_SOUTH = Tensor.ofFlat(1l, 0);
    static final Tensor<Long> STEP_NORTH = Tensor.ofFlat(-1l, 0);
    static final Tensor<Long> STEP_EAST = Tensor.ofFlat(0l, 1);
    static final Tensor<Long> STEP_WEST = Tensor.ofFlat(0l, -1);
    static final Tensor<Long> SCALAR_SHAPE = Tensor.ofFlat(new long[0]);
    static final Tensor<Long> WALL = Tensor.ofScalar('#');

    @CodeReflection
    public static Tensor<Long> turnLeft(Tensor<Long> direction) {
        return If(Equal(direction, DIRECTION_EAST),
                () -> Identity(DIRECTION_NORTH),
                () -> If(Equal(direction, DIRECTION_NORTH),
                        () -> Identity(DIRECTION_WEST),
                        () -> If(Equal(direction, DIRECTION_WEST),
                            () -> Identity(DIRECTION_SOUTH),
                            () -> Identity(DIRECTION_EAST))));
    }

    @CodeReflection
    public static Tensor<Long> turnRight(Tensor<Long> direction) {
        return Loop(THREE, TRUE, direction, (i, cond, d)
                -> LoopReturn(cond, turnLeft(d)));
    }

    @CodeReflection
    public static Tensor<Boolean> isWallAt(Tensor<Long> pos) {
        return Equal(CastLike(Slice(MAZE, pos, Add(pos, ONE_ONE), empty(), empty()), WALL, empty()), WALL);
    }

    @CodeReflection
    public static Tensor<Long> posInFrontOfMe(Tensor<Long> myPos, Tensor<Long> myDirection) {
        return  If(Equal(myDirection, DIRECTION_EAST),
                () -> Add(myPos, STEP_EAST),
                () -> If(Equal(myDirection, DIRECTION_NORTH),
                        () -> Add(myPos, STEP_NORTH),
                        () -> If(Equal(myDirection, DIRECTION_WEST),
                            () -> Add(myPos, STEP_WEST),
                            () -> Add(myPos, STEP_SOUTH))));
    }

    @CodeReflection
    public static Tensor<Boolean> atHome(Tensor<Long> pos) {
        return ReduceMin(Equal(pos, HOME_POS), empty(), empty(), empty());
    }

    @CodeReflection
    public static Tensor<Long> lastPos(Tensor<Long> pathLog) {
        return Slice(pathLog, MTHREE, MONE, empty(), empty());
    }

    @CodeReflection
    public static Tensor<Long> lastDirection(Tensor<Long> pathLog) {
        return Slice(pathLog, MONE, MAX, empty(), empty());
    }

    @CodeReflection
    public static Tensor<Long> addToLog(Tensor<Long> pathLog, Tensor<Long> pos, Tensor<Long> direction) {
        return Concat(List.of(pathLog, pos, direction), 0);
    }

    @CodeReflection
    public static Tensor<Byte> extractDirections(Tensor<Long> pathLog) {
        return Cast(Slice(pathLog, TWO, MAX, Optional.of(ZERO), Optional.of(THREE)), empty(), 3);
    }

    @CodeReflection
    public static Tensor<Long> turnLeftWhileWall(Tensor<Long> pos, Tensor<Long> direction) {
        var initialCond = Reshape(isWallAt(posInFrontOfMe(pos, direction)), SCALAR_SHAPE, empty());
        return Loop(LIMIT, initialCond, direction, (_, _, dir) -> {
                dir = turnLeft(dir);
                return LoopReturn(isWallAt(posInFrontOfMe(pos, dir)), dir);
            });
    }

    @CodeReflection
    public static Tensor<Long> walkAroundTheMaze() {
        var start = Concat(List.of(HOME_POS, DIRECTION_EAST), 0);
        var pathLog = Loop(LIMIT, TRUE, start, (_, _, log) -> {
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
        var directions = execute(() -> extractDirections(walkAroundTheMaze()));
        Assertions.assertEquals(EXPECTED_PATH, new String(directions.data().toArray(ValueLayout.JAVA_BYTE)));
    }
}
