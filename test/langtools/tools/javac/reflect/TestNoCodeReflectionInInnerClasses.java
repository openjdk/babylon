/*
 * @test /nodynamiccopyright/
 * @modules jdk.incubator.code
 * @compile/fail/ref=TestNoCodeReflectionInInnerClasses.out -XDrawDiagnostics TestNoCodeReflectionInInnerClasses.java
 */

import jdk.incubator.code.*;

class TestNoCodeReflectionInInnerClasses {
    class Inner {
        @CodeReflection
        public void test1() { }

        void test2() {
            Quotable q = (Runnable & Quotable) () -> { };
        }

        void test3() {
            Quoted q = () -> { };
        }

        void test4() {
            Quotable q = (Runnable & Quotable) this::test2;
        }
    }
}