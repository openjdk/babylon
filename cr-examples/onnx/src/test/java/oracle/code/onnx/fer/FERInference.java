package oracle.code.onnx.fer;

import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.net.URL;
import java.util.List;
import javax.imageio.ImageIO;

import oracle.code.onnx.foreign.coreml.coreml_provider_factory_h;
import oracle.code.onnx.foreign.OrtApi;

import static oracle.code.onnx.foreign.coreml.coreml_provider_factory_h_1.*;

public class FERInference {
    
    private static final String MODEL_PATH = "/oracle/code/onnx/fer/emotion-ferplus-8.onnx";
    private static final int IMAGE_SIZE = 64;
    
    private static final String[] EMOTIONS = {
            "neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"
    };
    
    private final OnnxRuntime runtime;
    
    public FERInference() throws Exception {
        runtime = OnnxRuntime.getInstance();
    }
    
    public float[] analyzeImage(URL imageUrl) throws Exception {
        try (var inferenceArena = Arena.ofConfined()) {
            var sessionOptions = runtime.createSessionOptions(inferenceArena);
            enableVerboseLogging(sessionOptions, inferenceArena);
            enableCoreML(sessionOptions, inferenceArena);
            
            URL modelUrl = FERInference.class.getResource(MODEL_PATH);
            if (modelUrl == null) {
                throw new RuntimeException("Model not found: " + MODEL_PATH);
            }
            byte[] modelBytes = modelUrl.openStream().readAllBytes();
            var inferenceSession = runtime.createSession(inferenceArena, modelBytes, sessionOptions);
            
            float[] imageData = loadImageAsFloatArray(imageUrl);
            
            long[] shape = {1, 1, IMAGE_SIZE, IMAGE_SIZE};
            var inputTensor = Tensor.ofShape(inferenceArena, shape, imageData);
            
            List<Tensor> outputs = inferenceSession.run(inferenceArena, List.of(inputTensor));
            
            float[] rawScores = outputs.get(0)
                    .data().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
            // model does not output softmax, so we need to apply it ourselves         
            float[] probs = softmax(rawScores);
            
            return probs;
        } catch (Exception e) {
            System.err.println("FERInference error for " + imageUrl + ": " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }
    
    private float[] loadImageAsFloatArray(URL imgUrl) throws IOException {
        BufferedImage src = ImageIO.read(imgUrl);
        if (src == null) {
            throw new IOException("Unsupported or corrupt image: " + imgUrl);
        }
        
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
    
    public static String[] getEmotions() {
        return EMOTIONS.clone();
    }
    
    private static void enableVerboseLogging(OnnxRuntime.SessionOptions sessionOptions, Arena arena) {
        var sessionOptionsAddress = getSessionOptionsAddress(sessionOptions);
        try {
            var runtimeField = OnnxRuntime.class.getDeclaredField("runtimeAddress");
            runtimeField.setAccessible(true);
            var runtimeAddress = (MemorySegment) runtimeField.get(OnnxRuntime.getInstance());

            var status = OrtApi.SetSessionLogSeverityLevel(runtimeAddress, sessionOptionsAddress, 0);

            if (status.address() == 0) {
                System.out.println("Verbose logging enabled (level 0)");
            } else {
                System.out.println("Failed to enable verbose logging (status: " + status.address() + ")");
            }

        } catch (Exception e) {
            System.out.println("Error enabling verbose logging: " + e.getMessage());
        }
    }

    private static void enableCoreML(OnnxRuntime.SessionOptions sessionOptions, Arena arena) {
        var sessionOptionsAddress = getSessionOptionsAddress(sessionOptions);
        try {
            int coremlFlags = coreml_provider_factory_h.COREML_FLAG_USE_CPU_AND_GPU();
            MemorySegment status = coreml_provider_factory_h
                    .OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptionsAddress, coremlFlags);

            if (status == null || status.address() == 0) {
                System.out.println("CoreML execution provider enabled successfully!");
            } else {
                System.out.println("CoreML EP returned status: " + status.address());
                status = coreml_provider_factory_h.OrtSessionOptionsAppendExecutionProvider_CoreML(
                        sessionOptionsAddress, coreml_provider_factory_h.COREML_FLAG_USE_CPU_ONLY());
                if (status == null || status.address() == 0) {
                    System.out.println("CoreML execution provider enabled with CPU_ONLY fallback!");
                } else {
                    System.out.println("CoreML EP failed with all flags - " + status.address());
                }
            }

        } catch (UnsatisfiedLinkError e) {
            System.out.println("CoreML execution provider is not available in the native ONNX Runtime library");
            throw new RuntimeException("CoreML execution provider is not available in the native ONNX Runtime library (symbol missing).", e);
        } catch (Throwable t) {
            System.out.println("Unexpected error while enabling CoreML EP: " + t.getMessage());
            throw new RuntimeException("Unexpected error while enabling CoreML EP: " + t.getMessage(), t);
        }
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
