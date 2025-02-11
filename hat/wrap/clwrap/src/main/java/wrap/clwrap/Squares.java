package wrap.clwrap;

import wrap.Wrap;

import java.io.IOException;
import java.lang.foreign.Arena;

public class Squares {


    public static void main(String[] args) throws IOException {
        try (var arena = Arena.ofConfined()) {
            CLPlatform.CLDevice[] selectedDevice = new CLPlatform.CLDevice[1];
            CLPlatform.platforms(arena).forEach(platform -> {
                System.out.println("Platform Name " + platform.platformName());
                platform.devices.forEach(device -> {
                    System.out.println("   Compute Units     " + device.computeUnits());
                    System.out.println("   Device Name       " + device.deviceName());
                    System.out.println("   Built In Kernels  " + device.builtInKernels());
                    selectedDevice[0] = device;
                });
            });
            var context = selectedDevice[0].createContext();
            var program = context.buildProgram("""
                    __kernel void squares(__global int* in,__global int* out ){
                        int gid = get_global_id(0);
                        out[gid] = in[gid]*in[gid];
                    }
                    """);
            var kernel = program.getKernel("squares");
            var in = Wrap.IntArr.of(arena,512);
            var out = Wrap.IntArr.of(arena,512);
            for (int i = 0; i < 512; i++) {
                in.set(i,i);
            }
            ComputeContext computeContext = new ComputeContext(arena,20);
            var inMem = computeContext.register(in.ptr());
            var outMem = computeContext.register(out.ptr());

            kernel.run(computeContext,512, inMem, outMem);
            for (int i = 0; i < 512; i++) {
                System.out.println(i + " " + out.get(i));
            }
        }
    }

}
