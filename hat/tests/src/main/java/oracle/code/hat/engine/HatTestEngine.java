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
package oracle.code.hat.engine;

import oracle.code.hat.annotation.HatTest;

import java.lang.annotation.Annotation;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

public class HatTestEngine {

    private static class Stats {
        int passed = 0;
        int failed = 0;
        public void incrementPassed() {
            passed++;
        }
        public void incrementFailed() {
            failed++;
        }

        public int getPassed() {
            return passed;
        }
        public int getFailed() {
            return failed;
        }

        @Override
        public String toString() {
            return String.format("passed: %d, failed: %d", passed, failed);
        }
    }

    static void main(String[] args) {
        System.out.println("HAT Engine Testing Framework");

        // We get a set of classes from the arguments
        if (args.length < 1) {
            throw new RuntimeException("No test classes provided");
        }
        String classNameToTest = args[0];
        System.out.println("Testing class " + classNameToTest);

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

            Object instance = testClass.getDeclaredConstructor().newInstance();

            StringBuilder builder = new StringBuilder();
            HatTestFormatter.appendClass(builder, classNameToTest);

            Stats stats = new Stats();

            for (Method method : methodToTest) {
                try {
                    HatTestFormatter.testing(builder, method.getName());
                    method.invoke(instance);
                    HatTestFormatter.ok(builder);
                    stats.incrementPassed();
                } catch (IllegalAccessException | InvocationTargetException e) {
                    e.printStackTrace();
                } catch (HatAssertionError e) {
                    HatTestFormatter.fail(builder);
                    stats.incrementFailed();
                }
            }
            System.out.println("");
            System.out.println(builder);
            System.out.println(stats);

        } catch (ClassNotFoundException | InvocationTargetException | InstantiationException | IllegalAccessException |
                 NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }
}
