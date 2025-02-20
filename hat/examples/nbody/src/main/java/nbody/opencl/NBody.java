package nbody.opencl;


import nbody.NBodyGLWindow;
import wrap.clwrap.CLPlatform;
import wrap.clwrap.CLWrapComputeContext;
import wrap.glwrap.GLTexture;

import java.io.IOException;
import java.lang.foreign.Arena;

public class NBody {
    public static class CLNBodyGLWindow extends NBodyGLWindow {
        final CLPlatform.CLDevice.CLContext.CLProgram.CLKernel kernel;
        final CLWrapComputeContext CLWrapComputeContext;
        final CLWrapComputeContext.MemorySegmentState vel;
        final CLWrapComputeContext.MemorySegmentState pos;


        public CLNBodyGLWindow( Arena arena, int width, int height, GLTexture particle, int bodyCount, Mode mode) {
            super( arena, width, height, particle, bodyCount, mode);
            this.CLWrapComputeContext = new CLWrapComputeContext(arena, 20);
            this.vel = CLWrapComputeContext.register(xyzVelFloatArr.ptr());
            this.pos = CLWrapComputeContext.register(xyzPosFloatArr.ptr());

            var platforms = CLPlatform.platforms(arena);
            System.out.println("platforms " + platforms.size());
            var platform = platforms.get(0);
            platform.devices.forEach(device -> {
                System.out.println("   Compute Units     " + device.computeUnits());
                System.out.println("   Device Name       " + device.deviceName());
                System.out.println("   Device Vendor       " + device.deviceVendor());
                System.out.println("   Built In Kernels  " + device.builtInKernels());
            });
            var device = platform.devices.get(0);
            System.out.println("   Compute Units     " + device.computeUnits());
            System.out.println("   Device Name       " + device.deviceName());
            System.out.println("   Device Vendor       " + device.deviceVendor());

            System.out.println("   Built In Kernels  " + device.builtInKernels());
            var context = device.createContext();
            String code = switch (mode) {
                case Mode.OpenCL -> """
                        __kernel void nbody( __global float *xyzPos ,__global float* xyzVel, float mass, float delT, float espSqr ){
                            int body = get_global_id(0);
                            int STRIDE=4;
                            int Xidx=0;
                            int Yidx=1;
                            int Zidx=2;
                            int bodyStride = body*STRIDE;
                            int bodyStrideX = bodyStride+Xidx;
                            int bodyStrideY = bodyStride+Yidx;
                            int bodyStrideZ = bodyStride+Zidx;
                            
                            float accx = 0.0;
                            float accy = 0.0;
                            float accz = 0.0;
                            float myPosx = xyzPos[bodyStrideX];
                            float myPosy = xyzPos[bodyStrideY];
                            float myPosz = xyzPos[bodyStrideZ];
                            for (int i = 0; i < get_global_size(0); i++) {
                                int iStride = i*STRIDE;
                                float dx = xyzPos[iStride+Xidx] - myPosx;
                                float dy = xyzPos[iStride+Yidx] - myPosy;
                                float dz = xyzPos[iStride+Zidx] - myPosz;
                                float invDist =  (float) 1.0/sqrt((float)((dx * dx) + (dy * dy) + (dz * dz) + espSqr));
                                float s = mass * invDist * invDist * invDist;
                                accx = accx + (s * dx);
                                accy = accy + (s * dy);
                                accz = accz + (s * dz);
                            }
                            accx = accx * delT;
                            accy = accy * delT;
                            accz = accz * delT;
                            xyzPos[bodyStrideX] = myPosx + (xyzVel[bodyStrideX] * delT) + (accx * 0.5 * delT);
                            xyzPos[bodyStrideY] = myPosy + (xyzVel[bodyStrideY] * delT) + (accy * 0.5 * delT);
                            xyzPos[bodyStrideZ] = myPosz + (xyzVel[bodyStrideZ] * delT) + (accz * 0.5 * delT);
                         
                            xyzVel[bodyStrideX] = xyzVel[bodyStrideX] + accx;
                            xyzVel[bodyStrideY] = xyzVel[bodyStrideY] + accy;
                            xyzVel[bodyStrideZ] = xyzVel[bodyStrideZ] + accz;
                            
                        }
                        """;
                case Mode.OpenCL4 -> """
                        __kernel void nbody( __global float4 *xyzPos ,__global float4* xyzVel, float mass, float delT, float espSqr ){
                            float4 acc = (0.0,0.0,0.0,0.0);
                            float4 myPos = xyzPos[get_global_id(0)];
                            float4 myVel = xyzVel[get_global_id(0)];
                            for (int i = 0; i < get_global_size(0); i++) {
                                   float4 delta =  xyzPos[i] - myPos;
                                   float invDist =  (float) 1.0/sqrt((float)((delta.x * delta.x) + (delta.y * delta.y) + (delta.z * delta.z) + espSqr));
                                   float s = mass * invDist * invDist * invDist;
                                   acc= acc + (s * delta);
                            }
                            acc = acc*delT;
                            myPos = myPos + (myVel * delT) + (acc * delT)/2;
                            myVel = myVel + acc;
                            xyzPos[get_global_id(0)] = myPos;
                            xyzVel[get_global_id(0)] = myVel;
                                          
                        }
                        """;
                default -> throw new IllegalStateException();
            };
            var program = context.buildProgram(code);
            kernel = program.getKernel("nbody");
        }


        @Override
        protected void moveBodies() {
            if (mode.equals(Mode.OpenCL4) || mode.equals(Mode.OpenCL)) {
                if (frameCount == 0) {
                    vel.copyToDevice = true;
                    pos.copyToDevice = true;
                } else {
                    vel.copyToDevice = false;
                    pos.copyToDevice = false;
                }
                vel.copyFromDevice = false;
                pos.copyFromDevice = true;

                kernel.run(CLWrapComputeContext, bodyCount, pos, vel, mass, delT, espSqr);
            } else {
                super.moveBodies();
            }
        }
    }

    public static void main(String[] args) throws IOException {
        int particleCount = args.length > 2 ? Integer.parseInt(args[2]) : 32768;
        NBodyGLWindow.Mode mode = NBodyGLWindow.Mode.of(args.length > 3 ? args[3] : NBodyGLWindow.Mode.OpenCL.toString());
        System.out.println("mode" + mode);
        try (var arena = mode.equals(NBodyGLWindow.Mode.JavaMT4) || mode.equals(NBodyGLWindow.Mode.JavaMT) ? Arena.ofShared() : Arena.ofConfined()) {
            var particleTexture = new GLTexture(arena, NBody.class.getResourceAsStream("/particle.png"));
            new CLNBodyGLWindow( arena, 1000, 1000, particleTexture, particleCount, mode).bindEvents().mainLoop();
        }
    }
}

