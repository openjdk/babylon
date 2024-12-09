package bldr;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Stream;

public class Capabilities {
    interface Probe{

    }
    public static abstract class Capability {
        final public String name;
        Capability(String name) {
            this.name=name;
        }
        public abstract boolean available();
    }
    public static abstract class CMakeCapability extends Capability{
        CMakeProbe cmakeProbe;
        CMakeCapability(String name) {
            super(name);
        }
        public  void setCmakeProbe(CMakeProbe cmakeProbe){
            this.cmakeProbe = cmakeProbe;
        }
    }

    public Map<String, Capability> capabilityMap = new HashMap<>();

    public static Capabilities of(Capability ... capabilities) {
        return new Capabilities(capabilities);
    }

    public Stream<Capability> capabilities() {
        return capabilityMap.values().stream();
    }
    public Stream<Capability> capabilities(Predicate<Capability> filter) {
        return capabilities().filter(filter);
    }

    public boolean capabilityIsAvailable(String name) {
        return capabilities().anyMatch(c-> c.name.equalsIgnoreCase(name));
    }

    private Capabilities(Capability ... capabilities){
        List.of(capabilities).forEach(capability ->
                capabilityMap.put(capability.name, capability)
        );
    }

    public static class OpenCL extends CMakeCapability {
        public static String includeDirKey  = "CMAKE_OpenCL_INCLUDE_DIR";
        public OpenCL() {
            super("OpenCL");
        }
        public static OpenCL of(){
            return new OpenCL();
        }

        @Override
        public boolean available() {
            return cmakeProbe.hasKey(includeDirKey);
        }

        Bldr.Dir includeDir(){
            return Bldr.Dir.of(Path.of(cmakeProbe.value(includeDirKey)));
        }
    }

    public static class OpenGL extends CMakeCapability {
        public static String includeDirKey  = "CMAKE_OPENGL_INCLUDE_DIR";
        public OpenGL() {
            super("OpenGL");
        }
        public static OpenGL of(){
            return new OpenGL();
        }
        @Override
        public boolean available() {
            return cmakeProbe.hasKey(includeDirKey);
        }
        Bldr.Dir includeDir(){
            return Bldr.Dir.of(Path.of(cmakeProbe.value(includeDirKey)));
        }
    }

    public static class HIP extends CMakeCapability {
        public HIP() {
            super("HIP");
        }
        public static HIP of(){
            return new HIP();
        }
        @Override
        public boolean available() {
            return false;
        }
    }
    public static class CUDA extends CMakeCapability {
        public static String sdkRootDirKey  = "CMAKE_CUDA_SDK_ROOT_DIR";
        public static String sdkRootDirNotFoundValue  = "CUDA_SDK_ROOT_DIR-NOTFOUND";
        public CUDA() {
            super("CUDA");
        }
        public static CUDA of(){
            return new CUDA();
        }
        @Override
        public boolean available() {
            return cmakeProbe.hasKey(sdkRootDirKey) && !cmakeProbe.value(sdkRootDirKey).equals(sdkRootDirNotFoundValue);
        }
    }



}
