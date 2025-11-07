/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.test.engine;

import hat.Accelerator;
import hat.backend.Backend;
import hat.test.annotation.HatTest;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

public class HATTestEngine {

    public static boolean DETAIL_ERROR_STACK_TRACE = false;

    public static final String TEST_REPORT_FILE_NAME = "test_report.txt";

    private static class Stats {
        int passed = 0;
        int failed = 0;
        int unsupported = 0;
        public void incrementPassed() {
            passed++;
        }
        public void incrementFailed() {
            failed++;
        }
        public void incrementUnsupported() { unsupported++; }

        public int getPassed() {
            return passed;
        }
        public int getFailed() {
            return failed;
        }
        public int getUnsupported() { return unsupported; }

        @Override
        public String toString() {
            return String.format("passed: %d, failed: %d, unsupported: %d", passed, failed, unsupported);
        }
    }

    private static void testMethod(StringBuilder builder, Method method, Stats stats, Object instance) {
        try {
            HATTestFormatter.testing(builder, method.getName());
            method.invoke(instance);
            HATTestFormatter.ok(builder);
            stats.incrementPassed();
        } catch (HATAssertionError e) {
            HATTestFormatter.fail(builder);
            stats.incrementFailed();
        } catch (HATExpectedFailureException failureException) {
            HATTestFormatter.expectedToFail(builder, failureException.getMessage());
            stats.incrementUnsupported();
        } catch (IllegalAccessException | InvocationTargetException e) {
            if (e.getCause() instanceof HATAssertionError hatAssertionError) {
                HATTestFormatter.failWithReason(builder, hatAssertionError.getMessage());
                stats.incrementFailed();
            } else if (e.getCause() instanceof HATExpectedFailureException failureException) {
                HATTestFormatter.expectedToFail(builder, failureException.getMessage());
                stats.incrementUnsupported();
            }  else {
                e.getCause().printStackTrace();
                HATTestFormatter.fail(builder);
                if (DETAIL_ERROR_STACK_TRACE) {
                    e.printStackTrace();
                }
                stats.incrementFailed();
            }
        }
    }

    private static void filterIfNeeded(String filterMethod, List<Method> methodsToTest) {
        // Filter the methodToTest if filterMethod is enabled
        List<Method> replaceMethods = new ArrayList<>();
        if (filterMethod != null) {
            for (Method declaredMethod : methodsToTest) {
                if (declaredMethod.getName().equals(filterMethod)) {
                    replaceMethods.add(declaredMethod);
                }
            }
            methodsToTest.clear();
            methodsToTest.addAll(replaceMethods);
        }
    }

    public static void printStats(StringBuilder builder, Stats stats) {
        System.out.println();
        System.out.println(builder.toString());
        System.out.println(stats.toString());
        System.out.println();
    }

    public static void dumpStats(StringBuilder builder, Stats stats) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(TEST_REPORT_FILE_NAME, true))){
            writer.write(builder.toString());
            writer.write(stats.toString());
            writer.newLine();
            writer.newLine();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void testClassEngine(String classNameToTest) {
        String filterMethod = null;
        if (classNameToTest.contains("#")) {
            String[] split = classNameToTest.split("#");
            classNameToTest = split[0];
            filterMethod = split[1];
        }

        try {
            Class<?> testClass = Class.forName(classNameToTest);
            // Obtain all method with the desired annotation
            List<Method> methodToTest = new ArrayList<>();
            for (Method declaredMethod : testClass.getDeclaredMethods()) {
                Annotation[] declaredAnnotations = declaredMethod.getDeclaredAnnotations();
                for (Annotation declaredAnnotation : declaredAnnotations) {
                    if (declaredAnnotation.annotationType().equals(HatTest.class)) {
                        methodToTest.add(declaredMethod);
                    }
                }
            }

            if (methodToTest.isEmpty()) {
                throw new RuntimeException("No test methods found for class " + classNameToTest);
            }

            filterIfNeeded(filterMethod, methodToTest);

            Object instance = testClass.getDeclaredConstructor().newInstance();

            StringBuilder builder = new StringBuilder();
            HATTestFormatter.appendClass(builder, classNameToTest);
            Stats stats = new Stats();

            for (Method method : methodToTest) {
                testMethod(builder, method, stats, instance);
            }
            printStats(builder, stats);
            dumpStats(builder, stats);

        } catch (ClassNotFoundException | InvocationTargetException | InstantiationException | IllegalAccessException |
                 NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }

    static void main(String[] args) {
        IO.println(Colours.BLUE + "HAT Engine Testing Framework" + Colours.RESET);
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        IO.println(Colours.BLUE + "Testing Backend: " + accelerator.backend.getName() + Colours.RESET);
        // We get a set of classes from the arguments
        if (args.length < 1) {
            throw new RuntimeException("No test classes provided");
        }
        testClassEngine(args[0]);
    }
}
