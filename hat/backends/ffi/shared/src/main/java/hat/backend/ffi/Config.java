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

package hat.backend.ffi;

import hat.codebuilders.HATCodeBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public record Config(int bits) {
    record Bit(int index, String name) {
    }

    // These must sync with hat/backends/ffi/opencl/include/opencl_backend.h
    // Bits 0-3 select platform id 0..5
    // Bits 4-7 select device id 0..15
    private static final int START_BIT_IDX = 16;
    private static final int MINIMIZE_COPIES_BIT = 1 << START_BIT_IDX;
    private static final int TRACE_BIT = 1 << 17;
    private static final int PROFILE_BIT = 1 << 18;
    private static final int SHOW_CODE_BIT = 1 << 19;
    private static final int SHOW_KERNEL_MODEL_BIT = 1 << 20;
    private static final int SHOW_COMPUTE_MODEL_BIT = 1 << 21;
    private static final int INFO_BIT = 1 << 22;
    private static final int TRACE_COPIES_BIT = 1 << 23;
    private static final int TRACE_SKIPPED_COPIES_BIT = 1 << 24;
    private static final int TRACE_ENQUEUES_BIT = 1 << 25;
    private static final int TRACE_CALLS_BIT = 1 << 26;
    private static final int SHOW_WHY_BIT = 1 << 27;
    private static final int SHOW_STATE_BIT = 1 << 28;
    private static final int PTX_BIT = 1 << 29;
    private static final int INTERPRET_BIT = 1 << 30;
    private static final int END_BIT_IDX = 31;

    private static String[] bitNames = {
            "MINIMIZE_COPIES",
            "TRACE",
            "PROFILE",
            "SHOW_CODE",
            "SHOW_KERNEL_MODEL",
            "SHOW_COMPUTE_MODEL",
            "INFO",
            "TRACE_COPIES",
            "TRACE_SKIPPED_COPIES",
            "TRACE_ENQUEUES",
            "TRACE_CALLS",
            "SHOW_WHY",
            "SHOW_STATE",
            "PTX",
            "INTERPRET",
    };

    public static Config of() {
        if (System.getenv("HAT") instanceof String opts) {
            System.out.println("From env " + opts);
            return of(opts);
        }
        if (System.getProperty("HAT") instanceof String opts) {
            System.out.println("From prop " + opts);
            return of(opts);
        }
        return of("");
    }

    public static Config of(int bits) {
        return new Config(bits);
    }

    public static Config of(List<Config> configs) {
        int allBits = 0;
        for (Config config : configs) {
            allBits |= config.bits;
        }
        return new Config(allBits);
    }

    public static Config of(Config... configs) {
        return of(List.of(configs));
    }

    public Config and(Config... configs) {
        return Config.of(Config.of(List.of(configs)).bits & bits);
    }

    public Config or(Config... configs) {
        return Config.of(Config.of(List.of(configs)).bits | bits);
    }

    public static Config of(String name) {
        if (name == null || name.equals("")) {
            return Config.of(0);
        }
        for (int i = 0; i < bitNames.length; i++) {
            if (bitNames[i].equals(name)) {
                return new Config(1 << (i + START_BIT_IDX));
            }
        }
        if (name.contains(",")) {
            List<Config> configs = new ArrayList<>();
            Arrays.stream(name.split(",")).forEach(opt ->
                    configs.add(of(opt))
            );
            return of(configs);
        } else if (name.contains(":")) {
            var tokens = name.split(":");
            if (tokens.length == 2) {
                var token = tokens[0];
                if (token.equals("PLATFORM") || token.equals("DEVICE")) {
                    int value = Integer.parseInt(tokens[1]);
                    return new Config(value << (token.equals("DEVICE") ? 4 : 0));
                } else {
                    System.out.println("Unexpected opt '" + name + "'");
                    return Config.of(0);
                }
            } else {
                System.out.println("Unexpected opt '" + name + "'");
                return Config.of(0);
            }
        } else {
            System.out.println("Unexpected opt '" + name + "'");
            System.exit(1);
            return Config.of(0);
        }
    }

    public static Config PTX() {
        return new Config(PTX_BIT);
    }

    public boolean isINTERPRET() {
        return (bits & INTERPRET_BIT) == INTERPRET_BIT;
    }

    public static Config INTERPRET() {
        return new Config(INTERPRET_BIT);
    }

    public boolean isPTX() {
        return (bits & PTX_BIT) == PTX_BIT;
    }

    public static Config SHOW_STATE() {
        return new Config(SHOW_STATE_BIT);
    }

    public boolean isSHOW_STATE() {
        return (bits & SHOW_STATE_BIT) == SHOW_STATE_BIT;
    }

    public static Config SHOW_WHY() {
        return new Config(SHOW_WHY_BIT);
    }

    public boolean isSHOW_WHY() {
        return (bits & SHOW_WHY_BIT) == SHOW_WHY_BIT;
    }

    public static Config TRACE_COPIES() {
        return new Config(TRACE_COPIES_BIT);
    }

    public boolean isTRACE_COPIES() {
        return (bits & TRACE_COPIES_BIT) == TRACE_COPIES_BIT;
    }

    public static Config TRACE_CALLS() {
        return new Config(TRACE_CALLS_BIT);
    }

    public boolean isTRACE_CALLS() {
        return (bits & TRACE_CALLS_BIT) == TRACE_CALLS_BIT;
    }

    public static Config TRACE_ENQUEUES() {
        return new Config(TRACE_ENQUEUES_BIT);
    }

    public boolean isTRACE_ENQUEUES() {
        return (bits & TRACE_ENQUEUES_BIT) == TRACE_ENQUEUES_BIT;
    }


    public static Config TRACE_SKIPPED_COPIES() {
        return new Config(TRACE_SKIPPED_COPIES_BIT);
    }

    public boolean isTRACE_SKIPPED_COPIES() {
        return (bits & TRACE_SKIPPED_COPIES_BIT) == TRACE_SKIPPED_COPIES_BIT;
    }

    public static Config INFO() {
        return new Config(INFO_BIT);
    }

    public boolean isINFO() {
        return (bits & INFO_BIT) == INFO_BIT;
    }


    public static Config PROFILE() {
        return new Config(PROFILE_BIT);
    }

    public boolean isPROFILE() {
        return (bits & PROFILE_BIT) == PROFILE_BIT;
    }

    public static Config TRACE() {
        return new Config(TRACE_BIT);
    }

    public boolean isTRACE() {
        return (bits & TRACE_BIT) == TRACE_BIT;
    }

    public static Config MINIMIZE_COPIES() {
        return new Config(MINIMIZE_COPIES_BIT);
    }

    public boolean isMINIMIZE_COPIES() {
        return (bits & MINIMIZE_COPIES_BIT) == MINIMIZE_COPIES_BIT;
    }

    public static Config SHOW_CODE() {
        return new Config(SHOW_CODE_BIT);
    }

    public boolean isSHOW_CODE() {
        return (bits & SHOW_CODE_BIT) == SHOW_CODE_BIT;
    }

    public static Config SHOW_KERNEL_MODEL() {
        return new Config(SHOW_KERNEL_MODEL_BIT);
    }

    public boolean isSHOW_KERNEL_MODEL() {
        return (bits & SHOW_KERNEL_MODEL_BIT) == SHOW_KERNEL_MODEL_BIT;
    }

    public static Config SHOW_COMPUTE_MODEL() {
        return new Config(SHOW_COMPUTE_MODEL_BIT);
    }

    public boolean isSHOW_COMPUTE_MODEL() {
        return (bits & SHOW_COMPUTE_MODEL_BIT) == SHOW_COMPUTE_MODEL_BIT;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (int bitIdx = START_BIT_IDX; bitIdx < END_BIT_IDX; bitIdx++) {
            if ((bits & (1 << bitIdx)) == (1 << bitIdx)) {
                if (!builder.isEmpty()) {
                    builder.append("|");
                }
                builder.append(bitNames[bitIdx - START_BIT_IDX]);

            }
        }
        return builder.toString();
    }

    public static class Main {
        public static class ConfigBuilder extends HATCodeBuilder<ConfigBuilder> {

            ConfigBuilder staticConstInt(String name, int padWidth, int value) {
                staticKeyword().space().constexprKeyword().space().intType().space().identifier(name, padWidth).space().equals().space().intHexValue(value).semicolon().nl();
                return this;
            }

            ConfigBuilder staticConstIntShiftedOne(String name, int padWidth, int shift) {
                staticKeyword().space().constexprKeyword().space().intType().space().identifier(name, padWidth).space().equals().space().intValue(1).leftShift().intHexValue(shift).semicolon().nl();
                return this;
            }

            ConfigBuilder className() {
                return identifier("BasicConfig");
            }

            ConfigBuilder bitNamesVar() {
                return identifier("bitNames");
            }

            ConfigBuilder configBitsVar() {
                return identifier("configBits");
            }

            ConfigBuilder configBitsAnd() {
                return configBitsVar().space().ampersand().space();
            }

            ConfigBuilder configBitsAndBitName(String bitName) {
                return configBitsAnd().identifier(bitName + "_BIT");
            }

            static String toCamelExceptFirst(String s) {
                String[] parts = s.split("_");
                StringBuilder camelCaseString = new StringBuilder("");
                for (String part : parts) {
                    camelCaseString.append(camelCaseString.isEmpty()
                            ? part.toLowerCase()
                            : part.substring(0, 1).toUpperCase() + part.substring(1).toLowerCase());
                }
                return camelCaseString.toString();
            }

            ConfigBuilder camelExceptFirst(String s) {
                return identifier(toCamelExceptFirst(s));
            }

            ConfigBuilder std(String s) {
                return identifier("std").colon().colon().identifier(s);
            }

            ConfigBuilder stdEndl() {
                return std("endl");
            }

            ConfigBuilder stdCout(String s) {
                return std("cout").space().leftShift().space().dquote().literal(s).dquote();
            }
        }

        public static void main(String[] args) {
            Config c = Config.of("INFO,PTX");
            ConfigBuilder cb = new ConfigBuilder();

            cb.lineComment("Auto generated from  " + Config.class.getName());
            cb.pragma("once").nl();
            cb.includeSys("iostream").nl();
            final int START_BIT_INDEX = 0x10;

            cb.structKeyword().space().className().braceNlIndented((_) -> {
                cb.staticConstInt("START_BIT_IDX", 32, START_BIT_INDEX);
                int i = START_BIT_INDEX;
                for (var bitname : Config.bitNames) {
                    cb.staticConstIntShiftedOne(bitname + "_BIT", 32, i++);
                }
                cb.staticConstInt("END_BIT_IDX", 32, i);
                cb.constKeyword().space().staticKeyword().space().charType().space().asterisk().bitNamesVar().osbrace().csbrace().semicolon().space().lineComment("See below for initialization");
                cb.intType().space().identifier("configBits").semicolon().nl();

                for (var bitName : Config.bitNames) {
                    cb.identifier("bool").space().camelExceptFirst(bitName).semicolon().nl();
                }

                cb.intType().space().identifier("platform").semicolon().nl();
                cb.intType().space().identifier("device").semicolon().nl();
                cb.identifier("bool").space().identifier("alwaysCopy").semicolon().nl();
//Constructor
                cb.explicitKeyword().space().className().paren((_) -> cb.intType().space().configBitsVar()).colon().nl().indent((_) -> {
                    cb.configBitsVar().paren((_) -> cb.configBitsVar()).comma().nl();
                    for (var bitName : Config.bitNames) {
                        cb.camelExceptFirst(bitName).paren((_) -> cb.paren((_) -> cb.configBitsAndBitName(bitName)).eq().identifier(bitName + "_BIT")).comma().nl();

                    }
                    cb.identifier("platform").paren((_) -> cb.configBitsAnd().intHexValue(0xf)).comma().nl();
                    cb.identifier("alwaysCopy").paren(_->cb.pling().camelExceptFirst("MINIMIZE_COPIES")).comma().nl();
                    cb.identifier("device").paren(_ ->
                            cb.paren(_ -> cb.configBitsAnd().intHexValue(0xf0)).space().rightShift().space().intValue(4)).braceNlIndented(_ ->
                            cb.ifKeyword().paren(_ -> cb.identifier("info")).braceNlIndented(_ -> {
                                for (var bitName : Config.bitNames) {
                                    cb.stdCout("native " + ConfigBuilder.toCamelExceptFirst(bitName) + " ").space().leftShift().space().camelExceptFirst(bitName).space().leftShift().space().stdEndl().semicolon().nl();
                                }
                            })
                    );
                }).nl().nl();

                cb.virtualKeyword().space().tilde().className().paren((_) -> {
                }).equals().space().defaultKeyword().semicolon();
            }).semicolon().nl().nl();


            cb.hashIfdef("shared_cpp", (_) -> {
                cb.constKeyword().space().charType().space().asterisk().className().colon().colon().bitNamesVar().sbrace(_ -> {}).equals().brace((_) -> {
                    cb.nl();
                    for (var bitName : Config.bitNames) {
                        cb.dquote().identifier(bitName + "_BIT").dquote().comma().nl();
                    }
                }).semicolon().nl();
            });
            System.out.println(cb);
        }
    }
}
