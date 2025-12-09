/*
 * @test /nodynamiccopyright/
 * @modules jdk.incubator.code
 * @compile/fail/ref=TestNoCodeReflectionInInnerClasses.out -Werror -Xlint:-incubating -XDrawDiagnostics TestNoCodeReflectionInInnerClasses.java
 */

import jdk.incubator.code.*;

class TestNoCodeReflectionInInnerClasses {
    class Inner {
        @Reflect
        public void test1() { }

        void test2() {
            Runnable q = (@Reflect Runnable) () -> { };
        }

        void test3() {
            Runnable q = (@Reflect Runnable) this::test2;
        }
    }
}
