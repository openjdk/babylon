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

package hat;

import java.util.List;

public class Config {

    public record Bit(int index, int size, String name, String description) implements Comparable<Bit> {
        static Bit of(int index, int size, String name, String description){
            return new Bit(index,size,name,description);
        }
        public static Bit of(int index, String name, String description){
            return new Bit(index,1,name,description);
        }

        public static Bit nextBit(Bit bit, int size, String name, String description){
            return new Bit(bit.index+bit.size,size,name, description);
        }
        public static Bit nextBit(Bit bit, String name, String description){
            return nextBit(bit, 1,name,description);
        }
        @Override
        public int compareTo(Bit bit) {
            return Integer.compare(index, bit.index);
        }

       public boolean isBitSet(int bits){
            return (mask()&bits) == mask();
        }
        public boolean isSet(Config config){
            return (mask()&config.bits) == mask();
        }
        public int mask(){
            return ((1<<size)-1) << index;
        }

        public String maskString(){
            return Integer.toBinaryString(mask());
        }
    }

    public static final Bit PLATFORM =  Bit.of(0,4, "PLATFORM", "FFI ONLY platform id (0-15)");
    public static final Bit DEVICE = Bit.nextBit(PLATFORM, 4, "DEVICE","FFI ONLY device id (0-15)");
    private static final Bit MINIMIZE_COPIES =  Bit.nextBit(DEVICE, "MINIMIZE_COPIES","FFI ONLY Try to minimize copies");
    public boolean minimizeCopies() {
        return MINIMIZE_COPIES.isSet(this);
    }
    public static final Bit TRACE = Bit.nextBit(MINIMIZE_COPIES,"TRACE", "FFI ONLY trace code");
    public static final Bit PROFILE = Bit.nextBit(TRACE, "PROFILE", "FFI ONLY Turn on profiling");
    private static final Bit SHOW_CODE = Bit.nextBit(PROFILE,"SHOW_CODE","Show generated code (PTX/OpenCL/CUDA)");
    public boolean showCode() {
        return SHOW_CODE.isSet(this);
    }
    private static final Bit SHOW_KERNEL_MODEL = Bit.nextBit(SHOW_CODE,"SHOW_KERNEL_MODEL", "Show (via OpWriter) Kernel Model");
    public boolean showKernelModel() {
        return SHOW_COMPUTE_MODEL.isSet(this);
    }
    private static final Bit SHOW_COMPUTE_MODEL = Bit.nextBit(SHOW_KERNEL_MODEL,"SHOW_COMPUTE_MODEL", "Show (via OpWriter) Compute Model");
    public boolean showComputeModel() {
        return SHOW_COMPUTE_MODEL.isSet(this);
    }
    public static final Bit INFO = Bit.nextBit(SHOW_COMPUTE_MODEL, "INFO", "FFI ONLY Show platform and device info");
    public static final Bit TRACE_COPIES = Bit.nextBit(INFO, "TRACE_COPIES", "FFI ONLY trace copies");
    public static final Bit TRACE_SKIPPED_COPIES = Bit.nextBit(TRACE_COPIES, "TRACE_SKIPPED_COPIES", "FFI ONLY Trace skipped copies (see MINIMIZE_COPIES) ");
    public static final Bit TRACE_ENQUEUES = Bit.nextBit(TRACE_SKIPPED_COPIES,"TRACE_ENQUEUES", "FFI ONLY trace enqueued tasks");
    public static final Bit TRACE_CALLS= Bit.nextBit(TRACE_ENQUEUES, "TRACE_CALLS", "FFI ONLY trace calls (enter/leave)");
    public static final Bit SHOW_WHY = Bit.nextBit(TRACE_CALLS, "SHOW_WHY", "FFI ONLY show why we decided to copy buffer (H to D)");
    public static final Bit SHOW_STATE = Bit.nextBit(SHOW_WHY, "SHOW_STATE", "Show iface buffer state changes");
    public static final Bit PTX = Bit.nextBit(SHOW_STATE, "PTX", "FFI (NVIDIA) ONLY pass PTX rather than C99 CUDA code");
    public static final Bit INTERPRET = Bit.nextBit(PTX, "INTERPRET", "Interpret the code model rather than converting to bytecode");
    private static final Bit NO_DIALECT = Bit.nextBit(INTERPRET, "NO_DIALECT", "Skip generating HAT dialect ops");
    public boolean interpret() {
        return INTERPRET.isSet(this);
    }
    private static final Bit HEADLESS = Bit.nextBit(NO_DIALECT, "HEADLESS", "Don't show UI");
    public boolean headless() {
        return HEADLESS.isSet(this)|| Boolean.getBoolean("headless");
    }
    public boolean headless(String arg) {
        return headless()|"--headless".equals(arg);
    }
    private static final Bit SHOW_LOWERED_KERNEL_MODEL = Bit.nextBit(HEADLESS,"SHOW_LOWERED_KERNEL_MODEL", "Show (via OpWriter) Lowered Kernel Model");
    public boolean showLoweredKernelModel() {
        return SHOW_LOWERED_KERNEL_MODEL.isSet(this);
    }
    private static final Bit SHOW_COMPILATION_PHASES = Bit.nextBit(SHOW_LOWERED_KERNEL_MODEL, "SHOW_COMPILATION_PHASES", "Show HAT compilation phases");
    private static final Bit PROFILE_CUDA_KERNEL = Bit.nextBit(SHOW_COMPILATION_PHASES, "PROFILE_CUDA_KERNEL", "Add -lineinfo to CUDA kernel compilation for profiling and debugging");
    public boolean showCompilationPhases() {
        return SHOW_COMPILATION_PHASES.isSet(this);
    }
    public boolean isProfileCUDAKernelEnabled() {
        return PROFILE_CUDA_KERNEL.isSet(this);
    }
    public static final List<Bit> bitList = List.of(
            PLATFORM,
            DEVICE,
            MINIMIZE_COPIES,
            TRACE,
            PROFILE,
            SHOW_CODE,
            SHOW_KERNEL_MODEL,
            SHOW_COMPUTE_MODEL,
            INFO,
            TRACE_COPIES,
            TRACE_SKIPPED_COPIES,
            TRACE_ENQUEUES,
            TRACE_CALLS,
            SHOW_WHY,
            SHOW_STATE,
            PTX,
            INTERPRET,
            NO_DIALECT,
            HEADLESS,
            SHOW_LOWERED_KERNEL_MODEL,
            SHOW_COMPILATION_PHASES,
            PROFILE_CUDA_KERNEL
    );

    private int bits;


    public int bits(){
        return bits;
    }
    public void bits(int bits){
        this.bits = bits;
    }

    Config(int bits){
        bits(bits);
    }

    // These must sync with hat/backends/ffi/shared/include/config.h
    // We can create the above config by running main() below...

    public static Config fromEnvOrProperty() {
        if (System.getenv("HAT") instanceof String opts) {
            System.out.println("From env " + opts);
            return fromSpec(opts);
        }
        if (System.getProperty("HAT") instanceof String opts) {
            System.out.println("From prop " + opts);
            return fromSpec(opts);
        }
        return fromSpec("");
    }

    public static Config fromIntBits(int bits) {
        return new Config(bits);
    }

    public static Config fromBits(List<Bit> configBits) {
        int allBits = 0;
        for (Bit configBit : configBits) {
            allBits |= configBit.mask();
        }
        return new Config(allBits);
    }

    public static Config fromBits(Bit... configBits) {
        return fromBits(List.of(configBits));
    }

    public Config and(Bit... configBits) {
        return Config.fromIntBits(Config.fromBits(List.of(configBits)).bits & bits);
    }

    public Config or(Bit... configBits) {
        return Config.fromIntBits(Config.fromBits(List.of(configBits)).bits | bits);
    }

    public record BitValue(Bit bit, int value){}

    public static Config fromSpec(String spec) {
        if (spec == null || spec.equals("")) {
            return Config.fromIntBits(0);
        }
        for (Bit bit:bitList) {
            if (bit.name().equals(spec)) {
                return new Config(bit.mask());
            }
        }
        if (spec.contains(",")) {
            var bits = 0;
            for (var opt: spec.split(",")) {
                var split = opt.split(":");
                var valName=split[0];
                var value=split.length==1?1:Integer.parseInt(split[1]);
                var bitValue = Config.bitList.stream()
                        .filter(bit ->bit.name().equals(valName))
                        .map(bit -> new BitValue(bit, value))
                        .findFirst()
                        .orElseThrow();
                bits |= bitValue.value << bitValue.bit.index();
            }
            return fromIntBits(bits);
        } else {
            System.out.println("Unexpected spec '" + spec + "'");
            System.exit(1);
            return Config.fromIntBits(0);
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (Bit bit:bitList){
            if (bit.isBitSet(bits)) {
                if (!builder.isEmpty()) {
                    builder.append("|");
                }
                builder.append(bit.name());

            }
        }
        return builder.toString();
    }

    public static void main(String[] args){
       bitList.forEach(b-> {
           System.out.printf("%30s MASK= %32s\n",  b.name,b.maskString());
       });

    }

}
