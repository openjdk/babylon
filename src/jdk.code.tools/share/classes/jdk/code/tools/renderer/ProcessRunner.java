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

package jdk.code.tools.renderer;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by gfrost
 */
public class ProcessRunner {

    static public ProcessRunner run(String prog) {
        return new ProcessRunner(prog);
    }

    static public ProcessRunner run(File file) {
        return new ProcessRunner(file);
    }

    List<String> args = new ArrayList<>();

    @SuppressWarnings("this-escape")
    public ProcessRunner(String prog) {
        opt(prog);
    }

    @SuppressWarnings("this-escape")
    public ProcessRunner(File prog) {
        file(prog);
    }

    public ProcessRunner opt(String... argsToAdd) {
        for (String s : argsToAdd) {
            args.add(s);
        }
        return this;
    }

    public ProcessRunner fileOpt(String opt, File file) {
        return opt(opt).file(file);
    }

    public ProcessRunner dirOpt(String opt, File dir) {
        dir.mkdir();
        return fileOpt(opt, dir);
    }

    public ProcessRunner file(File file) {
        opt(file.getAbsolutePath());
        return this;
    }

    public ProcessRunner files(File[] files) {
        for (File f : files) {
            file(f);
        }
        return this;
    }

    public void files(List<File> rsFiles) {
        files(rsFiles.toArray(new File[0]));
    }

    public static class Result {
        public Result() {
        }

        public void scan(Pattern pattern, Scanner scanner) {
            for (List<String> list : streams) { //stdout then stderr
                for (String text : list) {
                    Matcher matcher;
                    if ((matcher = pattern.matcher(text)).matches()) {
                        scanner.process(matcher);
                    }
                }
            }
        }

        public interface Scanner {
            void process(Matcher m);
        }

        public int status = -1;
        public boolean ok = false;
        public String commandLine = "";

        public List<String> stdout = new ArrayList<>();
        public List<String> stderr = new ArrayList<>();
        public List<List<String>> streams = List.of(stdout, stderr);
    }


    public Result go(boolean verbose) {
        Result result = new Result();

        StringBuilder commandBuilder = new StringBuilder();

        for (String arg : args) {
            commandBuilder.append(arg + " ");
            if (verbose) {
                System.out.print(arg + " ");
            }
        }

        result.commandLine = commandBuilder.toString();
        ProcessBuilder processBuilder = new ProcessBuilder(args);
        try {
            Process process = processBuilder.start();
            Thread stdout = new StreamReader("OUT", process.getErrorStream(), result.stdout, verbose).thread;
            Thread stderr = new StreamReader("ERR", process.getInputStream(), result.stderr, verbose).thread;
            result.status = process.waitFor();
            stdout.join();
            stderr.join();
            result.ok = result.status == 0;
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return result;
    }

    public ProcessRunner temp(String prefix, String suffix, String text) {
        try {
            File tempFile = File.createTempFile(prefix, suffix);
            FileWriter tempDotFileWriter = new FileWriter(tempFile);
            tempDotFileWriter.append(text);
            tempDotFileWriter.close();
            file(tempFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return this;
    }


    static class StreamReader {
        Thread thread;

        StreamReader(final String prefix, InputStream is, final List<String> out, final boolean verbose) {
            final BufferedReader br = new BufferedReader(new InputStreamReader(is));
            thread = new Thread(() -> {
                try {
                    for (String string = br.readLine(); string != null; string = br.readLine()) {
                        if (verbose) {
                            System.out.println(prefix + ":" + string);
                        }
                        out.add(string);
                    }
                    br.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
            thread.start();

        }

    }

}
