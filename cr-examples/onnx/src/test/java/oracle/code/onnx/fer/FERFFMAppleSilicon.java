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
import oracle.code.onnx.foreign.OrtApi;
//import oracle.code.onnx.foreign.onnxruntime_c_api_h;

import static oracle.code.onnx.foreign.coreml.coreml_provider_factory_h_1.*;

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
            
            // Check available providers first
            System.out.println("=== Available Execution Providers ===");
            checkAvailableProviders(runtime, arena);
            
            // Enable verbose logging to see execution provider usage
            System.out.println("\n=== Enabling Verbose Logging ===");
            enableVerboseLogging(sessionOptions, arena);
            
            // Explicitly enable CoreML execution provider
            System.out.println("\n=== Enabling CoreML Execution Provider ===");
            enableCoreML(sessionOptions, arena);
            
            var session = runtime.createSession(arena, modelBytes, sessionOptions);

            // Check which execution providers are being used
            System.out.println("\n=== Session Information ===");
            System.out.println("Session created successfully!");
            System.out.println("Model loaded and ready for inference.");
            System.out.println("CoreML execution provider has been explicitly enabled for GPU acceleration.");
            System.out.println("===========================\n");

            System.out.println("Model ready. Type an image name (without .png), or STOP to quit.");
            System.out.println("\n💡 TIP: To verify CoreML GPU usage, run this in another terminal:");
            System.out.println("   sudo powermetrics --samplers gpu_power -n 1 -i 1000");
            System.out.println("   (This will show GPU power usage during inference)\n");

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

                    // Perform inference with timing and verification
                    System.out.println("\n=== Running Inference ===");
                    long startTime = System.nanoTime();
                    
                    List<Tensor> outputs = session.run(arena, List.of(inputTensor));
                    
                    long endTime = System.nanoTime();
                    double inferenceTimeMs = (endTime - startTime) / 1_000_000.0;
                    
                    System.out.println("✓ Inference completed in " + String.format("%.2f", inferenceTimeMs) + " ms");
                    
                    // Performance analysis
                    if (inferenceTimeMs < 10) {
                        System.out.println("🚀 FAST inference - likely using GPU acceleration (CoreML)");
                    } else if (inferenceTimeMs < 50) {
                        System.out.println("⚡ MODERATE speed - may be using CoreML CPU or CPU fallback");
                    } else {
                        System.out.println("🐌 SLOW inference - likely using CPU-only execution");
                    }
                    
                    // Additional verification methods
                    System.out.println("\n=== CoreML Verification Methods ===");
                    System.out.println("1. Check verbose logs above for 'CoreML' execution messages");
                    System.out.println("2. Monitor GPU usage with: sudo powermetrics --samplers gpu_power -n 1 -i 1000");
                    System.out.println("3. Check Activity Monitor for GPU usage spikes during inference");
                    System.out.println("4. Fast inference times (<10ms) typically indicate GPU acceleration");
                    System.out.println("=====================================");
                    
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
     * Check which execution providers are available in the ONNX Runtime build.
     */
    private static void checkAvailableProviders(OnnxRuntime runtime, Arena arena) {
        try {
            // Get the runtime address using reflection
            var runtimeField = OnnxRuntime.class.getDeclaredField("runtimeAddress");
            runtimeField.setAccessible(true);
            var runtimeAddress = (MemorySegment) runtimeField.get(runtime);
            
            // Allocate memory for provider names and count
            var providersPtr = arena.allocate(C_POINTER);
            var countPtr = arena.allocate(C_INT);
            
            // Call GetAvailableProviders
            var status = oracle.code.onnx.foreign.OrtApi.GetAvailableProviders(runtimeAddress, providersPtr, countPtr);
            
            if (status.address() == 0) { // Success
                int count = countPtr.get(C_INT, 0);
                var providers = providersPtr.get(C_POINTER, 0);
                
                System.out.println("Found " + count + " available execution providers:");
                for (int i = 0; i < count; i++) {
                    var providerName = providers.getAtIndex(C_POINTER, i);
                    String name = providerName.getString(0);
                    System.out.println("  " + (i + 1) + ". " + name);
                }
                
                // Check if CoreML is available
                boolean coremlAvailable = false;
                for (int i = 0; i < count; i++) {
                    var providerName = providers.getAtIndex(C_POINTER, i);
                    String name = providerName.getString(0);
                    if (name.toLowerCase().contains("coreml")) {
                        coremlAvailable = true;
                        System.out.println("✓ CoreML execution provider is available!");
                        break;
                    }
                }
                
                if (!coremlAvailable) {
                    System.out.println("⚠ CoreML execution provider is NOT available in this build");
                }
                
            } else {
                System.out.println("Failed to get available providers (status: " + status.address() + ")");
            }
            
        } catch (Exception e) {
            System.out.println("Error checking available providers: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Enable verbose logging to see execution provider usage during inference.
     */
    private static void enableVerboseLogging(OnnxRuntime.SessionOptions sessionOptions, Arena arena) {
        var sessionOptionsAddress = getSessionOptionsAddress(sessionOptions);
        try {
            // Get the runtime address using reflection
            var runtimeField = OnnxRuntime.class.getDeclaredField("runtimeAddress");
            runtimeField.setAccessible(true);
            var runtimeAddress = (MemorySegment) runtimeField.get(OnnxRuntime.getInstance());
            
            // Set log severity level to VERBOSE (0)
            var status = oracle.code.onnx.foreign.OrtApi.SetSessionLogSeverityLevel(runtimeAddress, sessionOptionsAddress, 0);
            
            if (status.address() == 0) {
                System.out.println("✓ Verbose logging enabled (level 0)");
                System.out.println("  This will show detailed execution provider information during inference");
            } else {
                System.out.println("⚠ Failed to enable verbose logging (status: " + status.address() + ")");
            }
            
        } catch (Exception e) {
            System.out.println("✗ Error enabling verbose logging: " + e.getMessage());
        }
    }

    /**
     * Enable CoreML execution provider with optimal settings.
     */
    private static void enableCoreML(OnnxRuntime.SessionOptions sessionOptions, Arena arena) {
        var sessionOptionsAddress = getSessionOptionsAddress(sessionOptions);
        try {
            // Try to enable CoreML with GPU acceleration
            var method = coreml_provider_factory_h.class.getMethod(
                    "OrtSessionOptionsAppendExecutionProvider_CoreML",
                    MemorySegment.class, int.class
            );

            // Use CPU_AND_GPU flag for optimal performance on Apple Silicon
            int coremlFlags = coreml_provider_factory_h.COREML_FLAG_USE_CPU_AND_GPU();
            System.out.println("Enabling CoreML with CPU_AND_GPU flag (" + coremlFlags + ")");
            
            Object status = method.invoke(null, sessionOptionsAddress, coremlFlags);
            
            if (status == null || (status instanceof MemorySegment ms && ms.address() == 0)) {
                System.out.println("✓ CoreML execution provider enabled successfully!");
            } else {
                System.out.println("⚠ CoreML EP returned status: " + status);
                // Try fallback with CPU only
                System.out.println("Trying fallback with CPU_ONLY flag...");
                status = method.invoke(null, sessionOptionsAddress, coreml_provider_factory_h.COREML_FLAG_USE_CPU_ONLY());
                if (status == null || (status instanceof MemorySegment ms && ms.address() == 0)) {
                    System.out.println("✓ CoreML execution provider enabled with CPU_ONLY fallback!");
                } else {
                    System.out.println("✗ CoreML EP failed with all flags - " + status);
                }
            }

        } catch (NoSuchMethodException e) {
            System.out.println("✗ CoreML execution provider is not available in this ONNX Runtime build");
            throw new RuntimeException("CoreML execution provider is not available in this ONNX Runtime build (method missing in bindings).", e);
        } catch (UnsatisfiedLinkError e) {
            System.out.println("✗ CoreML execution provider is not available in the native ONNX Runtime library");
            throw new RuntimeException("CoreML execution provider is not available in the native ONNX Runtime library (symbol missing).", e);
        } catch (Throwable t) {
            System.out.println("✗ Unexpected error while enabling CoreML EP: " + t.getMessage());
            throw new RuntimeException("Unexpected error while enabling CoreML EP: " + t.getMessage(), t);
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

}