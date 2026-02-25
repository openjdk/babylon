/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
package hat.examples.common;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.stream.IntStream;

public class StatUtils {

    public static double computeAverage(List<Long> timers, int discard) {
        double sum = timers.stream().skip(discard).reduce(0L, Long::sum).doubleValue();
        int totalCountedValues = timers.size() - discard;
        return (sum / totalCountedValues);
    }

    public static double computeSpeedup(double baseline, double measured) {
        return (Math.ceil(baseline / measured * 100) / 100);
    }

    public static void printCheckResult(boolean isCorrect, String version) {
        IO.println(isCorrect ? version + " is correct" : version + " is wrong");
    }

    public static void dumpStatsToCSVFile(List<List<Long>> listOfTimers, List<String> header, final String fileName) {
        final int numColumns = listOfTimers.size();
        if (numColumns != header.size()) {
            throw new HATExampleException("Header size and List of timers need to be the same size");
        }
        StringBuilder builder = new StringBuilder();
        IntStream.range(0, header.size()).forEach(i -> {
            builder.append(header.get(i));
            if (i != header.size() - 1) {
                builder.append(",");
            }
        });
        builder.append(System.lineSeparator());

        final int numRows = listOfTimers.getFirst().size();
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numColumns; col++) {
                // all lists must be of the same size:
                if (listOfTimers.get(col).size() != numRows) {
                    throw new HATExampleException("[ERROR] Result List: " + col + " has a different size");
                }
                Long timer = listOfTimers.get(col).get(row);
                builder.append(timer);
                if (col != header.size() - 1) {
                    builder.append(",");
                }
            }
            builder.append(System.lineSeparator());
        }
        builder.append(System.lineSeparator());

        IO.println("[INFO] Saving results into file: " + fileName);
        try(BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            writer.append(builder.toString());
        } catch (IOException e) {
            throw new HATExampleException(e.getMessage());
        }
    }

    public static void sleep10Sec() {
        try {
            Thread.sleep(10_000);
        } catch (InterruptedException e) {
            throw new HATExampleException(e.getMessage());
        }
    }

    private StatUtils() {}
}
