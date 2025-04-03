package hat.backend.ffi;

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
    private static final int END_BIT_IDX = 29;

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
    };

    public static Config of() {
        if (System.getenv("HAT") instanceof String opts){
            System.out.println("From env "+opts);
            return of(opts);
        }
        if (System.getProperty("HAT") instanceof String opts) {
            System.out.println("From prop "+opts);
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
        if (name == null || name.equals("")){
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
        }else if (name.contains(":")){
            var tokens=name.split(":");
            if (tokens.length == 2) {
                var token = tokens[0];
                if (token.equals("PLATFORM") || token.equals("DEVICE")) {
                    int value = Integer.parseInt(tokens[1]);
                    return new Config(value<<(token.equals("DEVICE")?4:0));
                }else{
                    System.out.println("Unexpected opt '" + name + "'");
                    return Config.of(0);
                }
            }else{
                System.out.println("Unexpected opt '" + name + "'");
                return Config.of(0);
            }
        } else {
            System.out.println("Unexpected opt '" + name + "'");
            System.exit(1);
            return Config.of(0);
        }
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
}
