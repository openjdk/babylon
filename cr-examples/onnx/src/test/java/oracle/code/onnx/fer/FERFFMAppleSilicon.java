package oracle.code.onnx.fer;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Scanner;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.foreign.coreml.coreml_provider_factory_h;
//import oracle.code.onnx.foreign.onnxruntime_c_api_h;

public class FERFFMAppleSilicon {

    static final String BASE_PATH = "/oracle/code/onnx/fer/";
    static final String MODEL_PATH = BASE_PATH + "emotion-ferplus-8.onnx";
    static final int IMAGE_SIZE = 64;

    static final String[] EMOTIONS = {
            "neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"
    };

    public static void main(String[] args) {
        try (Scanner scanner = new Scanner(System.in);
             Arena arena = Arena.ofConfined()) {

            var runtime = OnnxRuntime.getInstance();

            var modelUrl = FERFFMAppleSilicon.class.getResource(MODEL_PATH);
            if (modelUrl == null) {
                throw new RuntimeException("Model not found: " + MODEL_PATH);
            }
            byte[] modelBytes;
            try {
                modelBytes = modelUrl.openStream().readAllBytes();
            } catch (IOException e) {
                throw new RuntimeException("Failed to load model: " + e.getMessage(), e);
            }

            var sessionOptions = runtime.createSessionOptions(arena);
            var sessionOptionsAddress = getSessionOptionsAddress(sessionOptions);

            // Hard requirement: must have CoreML, otherwise abort
            requireCoreML(sessionOptionsAddress);

            var session = createSessionWithOptions(runtime, arena, modelBytes, sessionOptionsAddress);

            System.out.println("\nModel ready. Type an image name (without .png), or STOP to quit.\n");

            while (true) {
                System.out.print("Image name: ");
                String name = scanner.nextLine().trim();
                if (name.equalsIgnoreCase("STOP")) {
                    System.out.println("Exiting.");
                    break;
                }

                var imgUrl = FERFFMAppleSilicon.class.getResource(BASE_PATH + name + ".png");
                if (imgUrl == null) {
                    System.out.println("File not found: " + BASE_PATH + name + ".png\n");
                    continue;
                }

                try {
                    float[] imageData = loadImageAsFloatArray(imgUrl);

                    long[] shape = {1, 1, IMAGE_SIZE, IMAGE_SIZE};
                    var inputTensor = Tensor.ofShape(arena, shape, imageData);

                    List<Tensor> outputs = session.run(arena, List.of(inputTensor));
                    var output = outputs.get(0);

                    float[] rawScores = output.data().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
                    float[] probs = softmax(rawScores);

                    System.out.println("\nEmotion probabilities:");
                    for (int i = 0; i < probs.length; i++) {
                        System.out.printf("%-10s : %.2f%%%n", EMOTIONS[i], probs[i] * 100);
                    }

                    int pred = argMax(probs);
                    System.out.println("Predicted emotion: " + EMOTIONS[pred] + "\n");

                } catch (IOException e) {
                    System.out.println("Error reading image: " + e.getMessage());
                }
            }
        }
    }

    // ---------- CoreML Setup ----------

    /**
     * Enable CoreML execution provider.
     * Throws RuntimeException if CoreML is not available.
     */
    private static void requireCoreML(MemorySegment sessionOptionsAddress) {
        try {
            // Look for the CoreML append function in the generated bindings
            var method = coreml_provider_factory_h.class.getMethod(
                    "OrtSessionOptionsAppendExecutionProvider_CoreML",
                    MemorySegment.class, int.class
            );

            // Call it with flags = 0
            Object status = method.invoke(null, sessionOptionsAddress, 0);

            if (status != null) {
                throw new RuntimeException("Failed to enable CoreML EP: " + status);
            }

            System.out.println("✓ Using CoreML execution provider for Apple Silicon GPU acceleration");

        } catch (NoSuchMethodException e) {
            throw new RuntimeException(
                    "CoreML execution provider is not available in this ONNX Runtime build (method missing in bindings).", e
            );
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException(
                    "CoreML execution provider is not available in the native ONNX Runtime library (symbol missing).", e
            );
        } catch (Throwable t) {
            throw new RuntimeException(
                    "Unexpected error while enabling CoreML EP: " + t.getMessage(), t
            );
        }
    }

    // ---------- Utility Methods ----------

    private static float[] loadImageAsFloatArray(java.net.URL imgUrl) throws IOException {
        BufferedImage src = ImageIO.read(imgUrl);
        if (src == null) throw new IOException("Unsupported or corrupt image: " + imgUrl);

        BufferedImage graySrc = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g0 = graySrc.createGraphics();
        g0.drawImage(src, 0, 0, null);
        g0.dispose();

        BufferedImage gray = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gray.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(graySrc, 0, 0, IMAGE_SIZE, IMAGE_SIZE, null);
        g.dispose();

        float[] data = new float[IMAGE_SIZE * IMAGE_SIZE];
        gray.getData().getSamples(0, 0, IMAGE_SIZE, IMAGE_SIZE, 0, data);

        return data;
    }

    private static float[] softmax(float[] scores) {
        float max = Float.NEGATIVE_INFINITY;
        for (float s : scores) if (s > max) max = s;
        double sum = 0.0;
        double[] exps = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            exps[i] = Math.exp(scores[i] - max);
            sum += exps[i];
        }
        float[] out = new float[scores.length];
        for (int i = 0; i < scores.length; i++) {
            out[i] = (float)(exps[i] / sum);
        }
        return out;
    }

    private static int argMax(float[] arr) {
        int idx = 0;
        float max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }

    private static MemorySegment getSessionOptionsAddress(OnnxRuntime.SessionOptions options) {
        try {
            var field = OnnxRuntime.SessionOptions.class.getDeclaredField("sessionOptionsAddress");
            field.setAccessible(true);
            return (MemorySegment) field.get(options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static OnnxRuntime.Session createSessionWithOptions(OnnxRuntime runtime,
                                                                Arena arena,
                                                                byte[] modelBytes,
                                                                MemorySegment sessionOptionsAddress) {
        try {
            var method = OnnxRuntime.class.getDeclaredMethod("createSession", Arena.class, byte[].class, OnnxRuntime.SessionOptions.class);
            method.setAccessible(true);
            var constructor = OnnxRuntime.SessionOptions.class.getDeclaredConstructor(MemorySegment.class);
            constructor.setAccessible(true);
            var sessionOptions = constructor.newInstance(sessionOptionsAddress);
            return (OnnxRuntime.Session) method.invoke(runtime, arena, modelBytes, sessionOptions);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}