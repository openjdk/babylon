package nbody;

public enum Mode {
    HAT, OpenCL, Cuda, OpenCL4, Cuda4, JavaSeq, JavaMT, JavaSeq4, JavaMT4;

    public static Mode of(String s) {
        return switch (s) {
            case "HAT" -> Mode.HAT;
            case "OpenCL" -> Mode.OpenCL;
            case "Cuda" -> Mode.Cuda;
            case "JavaSeq" -> Mode.JavaSeq;
            case "JavaMT" -> Mode.JavaMT;
            case "JavaSeq4" -> Mode.JavaSeq4;
            case "JavaMT4" -> Mode.JavaMT4;
            case "OpenCL4" -> Mode.OpenCL4;
            case "Cuda4" -> Mode.Cuda4;
            default -> throw new IllegalStateException("No mode " + s);
        };
    }
}
