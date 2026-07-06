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
package hat.test.engine;

import optkl.textmodel.terminal.ANSI;
import optkl.util.Regex;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class TestSuite {
    static int exec(Path dir, List<String> opts, Consumer<String> stdout, Consumer<String> stderr) {
        try {
            var process = new ProcessBuilder().command(opts).directory(dir.toFile()).start();
            new Thread(() -> new BufferedReader(new InputStreamReader(process.getInputStream())).lines().forEach(stdout)).start();
            new Thread(() -> new BufferedReader(new InputStreamReader(process.getErrorStream())).lines().forEach(stderr)).start();
            return process.waitFor();
        } catch (InterruptedException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    void main(String[] argArr) {
        var args = List.of(argArr);
        if (args.isEmpty()){
            throw new RuntimeException("Expecting backend (ffi-opencl | ffi-cuda");
        }
        var backend = args.get(0);
        var ansi = ANSI.of(System.out);
        var hatDir = Paths.get(System.getProperty("user.dir"));
        var testDir = hatDir.resolve("tests/src/main/java/hat/test/");
        var testPattern = Regex.of("^.*(Test[a-zA-Z0-9]*)\\.java$");
        List<String> tests = null;
        try {
            tests = Files.list(testDir)
                    .map(Path::toString)
                    .map(testPattern::is)
                    .filter(Regex.Match.class::isInstance)
                    .map(m->((Regex.Match)m).stringOf(1))
                    .toList();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        ansi.apply("""
                *****************************************************************
                HAT Test Report
                *****************************************************************
                """);

        var test_reports_txt = hatDir.resolve("test_report.txt");
        try {
            Files.deleteIfExists(test_reports_txt);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        var num_tests = hatDir.resolve(".num_tests");
        try {
            Files.deleteIfExists(num_tests);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        var commonOpts = List.of(
                "java",
                "--enable-preview",
                "--add-modules=jdk.incubator.code",
                "--enable-native-access=ALL-UNNAMED",
                "-Djava.library.path=build",
                "--class-path", "build/hat-optkl-1.0.jar:build/hat-core-1.0.jar:build/hat-backend-ffi-shared-1.0.jar:build/hat-backend-" + backend + "-1.0.jar:build/hat-tests-1.0.jar",
                "hat.test.engine.HATTestEngine"
        );

        class Stats {
            int count = 0;
            int passed = 0;
            int failed = 0;
            int unsupported = 0;
            int precisionError = 0;
            int total = 0;
            float passRate;

            public void update(int passed,int failed, int unsupported,int precisionError){
                this.passed+=passed;
                this.failed+=failed;
                this.unsupported+=unsupported;
                this.precisionError+=precisionError;
                this.total = this.passed + this.failed + this.unsupported + this.precisionError;
                this.passRate= this.passed>0f?this.passed * 100f / this.total:0f;
            }
        }
        var stats = new Stats();

        tests.forEach(test -> {
            var testOpts = new ArrayList<>(commonOpts);
            testOpts.add("hat.test." + test);
            testOpts.add("--count-tests");
            if (exec(hatDir, testOpts, stdout -> IO.println("OUT:" + stdout), _ -> {})  == 0) {
                try {
                    stats.count += Integer.parseInt(Files.readString(num_tests));
                } catch (IOException e) {
                    ansi.fail("failed to collect  " + stats.count + " for " + test+"\n");
                }
            } else {
                ansi.fail( "failed to run test " + test+"\n");
            }
        });
        var regex = Regex.of("passed: (\\d+), failed: (\\d+), unsupported: (\\d+), precision-errors: (\\d+)");

        tests.forEach(test -> {
            var testOpts = new ArrayList<>(commonOpts);
            testOpts.add("hat.test." + test);
            if (exec(hatDir, testOpts, stdout -> IO.println("   " + stdout), _ -> {}) != 0) {
                ansi.fail("failed to run test " + test+"\n");
            }
        });

        try {
            Files.readAllLines(test_reports_txt).forEach(line -> {
                if (regex.is(line) instanceof Regex.Match match) {
                    stats.update(match.intOf(1),match.intOf(2),match.intOf(3),match.intOf(4));
                }
            });
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        ansi.apply("Global ")
                .pass("passed: "+stats.passed+", ")
                .fail("failed: "+stats.failed+", ")
                .fail("unsupported: "+stats.unsupported+", ")
                .warn("precision-errors: "+stats.precisionError+", ")
                .apply("pass-rate: "+String.format("%.2f\n",stats.passRate));

        if (stats.total == stats.count) {
            ansi.pass( "[REPORT] OK: All tests launched. Total: " + stats.count+"\n");
        } else {
            ansi.fail(
                    "[REPORT] Test failed. Some tests were not launched. Common reasons: seg-faults, driver-issues. Please, check again.\n"+
                    "    - Expected to run: " + stats.count + ". But only launched " + stats.total+"\n");
        }

        if (stats.failed > 0) {
            System.exit(-1);
        } else {
            System.exit(0);
        }
    }
}
