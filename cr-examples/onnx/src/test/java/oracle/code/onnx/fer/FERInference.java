package oracle.code.onnx.fer;

import oracle.code.onnx.coreml.OnnxProvider;
import oracle.code.onnx.coreml.OnnxRuntime;
import oracle.code.onnx.coreml.Tensor;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.net.URL;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

public class FERInference {

	private static final Logger logger = Logger.getLogger(FERInference.class.getName());
	private static final String MODEL_PATH = "/oracle/code/onnx/fer/emotion-ferplus-8.onnx";
	private static final int IMAGE_SIZE = 64;

	private static final String[] EMOTIONS = {
			"neutral", "happiness", "surprise", "sadness",
			"anger", "disgust", "fear", "contempt"
	};

	private final OnnxRuntime runtime;

	public FERInference() {
		runtime = OnnxRuntime.getInstance();
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
			out[i] = (float) (exps[i] / sum);
		}
		return out;
	}

	public float[] analyzeImage(URL imageUrl) {
		try (var inferenceArena = Arena.ofConfined()) {
			var sessionOptions = runtime.createSessionOptions(inferenceArena);

			Map<String, String> options = Map.of("ModelFormat", "MLProgram",
					"MLComputeUnits", "CPUAndGPU", "EnableOnSubgraphs", "1",
					"AllowLowPrecisionAccumulationOnGPU", "1",
					"ModelCacheDirectory", Path.of(imageUrl.toURI()).getParent().toString());

			OnnxProvider provider = new OnnxProvider("CoreML", options);
			runtime.appendExecutionProvider(inferenceArena, sessionOptions, provider);

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

			float[] rawScores = outputs.getFirst()
					.data().toArray(ValueLayout.JAVA_FLOAT);
			// model does not output softmax, so we need to apply it ourselves
			float[] probs = softmax(rawScores);

			return probs;
		} catch (Exception e) {
			String errorMessage = "FERInference error for %s".formatted(imageUrl);
			logger.log(Level.SEVERE, errorMessage, e);
			throw new RuntimeException(errorMessage);
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
}
