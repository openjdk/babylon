package oracle.code.onnx;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.util.List;
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
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            #                                             #
            ###############################################
            """.getBytes());

    static final Tensor<Long> DIRECTION_NORTH = Tensor.ofFlat('N');
    static final Tensor<Long> DIRECTION_SOUTH = Tensor.ofFlat('S');
    static final Tensor<Long> DIRECTION_EAST = Tensor.ofFlat('E');
    static final Tensor<Long> DIRECTION_WEST = Tensor.ofFlat('W');

    static final Tensor<Long> HOME_POS = Tensor.ofFlat(20l, 1);

    static final Tensor<Boolean> TRUE = Tensor.ofScalar(true);
    static final Tensor<Boolean> FALSE = Tensor.ofScalar(false);
    static final Tensor<Long> ONE_ONE = Tensor.ofFlat(1l, 1);
    static final Tensor<Long> ZERO = Tensor.ofFlat(0l);
    static final Tensor<Long> TWO = Tensor.ofFlat(2l);
    static final Tensor<Long> THREE = Tensor.ofFlat(3l);
    static final Tensor<Long> MONE = Tensor.ofFlat(-1l);
    static final Tensor<Long> MTWO = Tensor.ofFlat(-2l);
    static final Tensor<Long> MTHREE = Tensor.ofFlat(-3l);
    static final Tensor<Long> MAX = Tensor.ofFlat(Long.MAX_VALUE);
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
    public static Tensor<Long> turnLeftWhileWall(Tensor<Long> pos, Tensor<Long> direction) {
        var initialCond = Reshape(isWallAt(posInFrontOfMe(pos, direction)), SCALAR_SHAPE, empty());
        return Loop(MAX, initialCond, direction, (_, _, newDirection) -> {
                return LoopReturn(isWallAt(posInFrontOfMe(pos, newDirection)), turnLeft(newDirection));
            });
    }


    @CodeReflection
    public static Tensor<Boolean> atHome(Tensor<Long> pos) {
        return ReduceMin(Equal(pos, HOME_POS), empty(), empty(), empty());
    }

    @CodeReflection
    public static Tensor<Long> walkAroundTheMaze() {
        var start = Concat(List.of(HOME_POS, DIRECTION_EAST), 0);
        return Loop(MAX, TRUE, start, (_, _, path) -> {
            var pos = Slice(path, MTHREE, MONE, empty(), empty());
            var direction = Slice(path, MONE, MAX, empty(), empty());

            var posInFront = posInFrontOfMe(pos, direction);

            var newPos = If(isWallAt(posInFront),
                    () -> Identity(pos),
                    () -> Identity(posInFront));

            var newDirection =  If(isWallAt(posInFront),
                    () -> turnLeft(direction),
                    () -> Identity(direction));
            return LoopReturn(Not(atHome(newPos)), Concat(List.of(path, newPos, newDirection), 0));
        });
    }

//    @CodeReflection
//    public static Tensor<Long> walkAroundTheMaze2() {
//        var start = Concat(List.of(HOME_POS, DIRECTION_EAST), 0);
//        return Loop(MAX, TRUE, start, (_, _, path) -> {
//            var pos = Slice(path, MTHREE, MONE, empty(), empty());
//            var direction = Slice(path, MONE, MAX, empty(), empty());
//            pos = posInFrontOfMe(pos, direction);
//            direction = turnRight(direction);
//            direction = turnLeftWhileWall(pos, direction);
//            return LoopReturn(Not(atHome(pos)), Concat(List.of(path, pos, direction), 0));
//        });
//    }

    @Test
    public void testWalkAroundTheMaze() throws Exception {
//        for (long y = 0; y < 22; y++) {
//            for (long x = 0; x < 47; x++) {
//                var pos = Tensor.ofFlat(y, x);
//                System.out.print(execute(() -> isWallAt(pos)).data().get(ValueLayout.JAVA_BOOLEAN, 0) ? "#" : " ");
//            }
//            System.out.println();
//        }

//        var dir = execute(() -> turnLeftWhileWall(HOME_POS, DIRECTION_WEST));
//        System.out.println((char)dir.data().get(ValueLayout.JAVA_LONG, 0));

        var path = execute(() -> walkAroundTheMaze());
        path.data().elements(MemoryLayout.sequenceLayout(3, ValueLayout.JAVA_LONG)).forEach(ms ->
                System.out.println("x: " + ms.getAtIndex(ValueLayout.JAVA_LONG, 1) + " y: " + ms.getAtIndex(ValueLayout.JAVA_LONG, 0) + " direction: " + (char)ms.getAtIndex(ValueLayout.JAVA_LONG, 2)));
    }

}
