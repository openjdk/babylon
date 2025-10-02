package oracle.code.onnx.fer;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.util.List;
import java.util.Scanner;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.RenderingHints;

import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;

public class FERFFMOnlyDemo {

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

			var modelUrl = FERFFMOnlyDemo.class.getResource(MODEL_PATH);
			if (modelUrl == null) {
				throw new RuntimeException("Model not found: " + MODEL_PATH);
			}
			byte[] modelBytes;
			try (InputStream urlStream = modelUrl.openStream()) {
				modelBytes = urlStream.readAllBytes();
			} catch (IOException e) {
				throw new RuntimeException("Failed to load model: " + e.getMessage(), e);
			}
			var session = runtime.createSession(arena, modelBytes);

			IO.println("\nModel ready.");

			while (true) {
				IO.print("Type Image name (without .png), or STOP to quit: ");
				String name = scanner.nextLine().trim();
				if (name.equalsIgnoreCase("STOP")) {
					IO.println("Exiting.");
					break;
				}

				var imgUrl = FERFFMOnlyDemo.class.getResource(BASE_PATH + name + ".png");
				if (imgUrl == null) {
					IO.println("File not found: " + BASE_PATH + name + ".png\n");
					continue;
				}

				try {
					float[] imageData = loadImageAsFloatArray(imgUrl);

					long[] shape = {1, 1, IMAGE_SIZE, IMAGE_SIZE};
					var inputTensor = Tensor.ofShape(arena, shape, imageData);

					List<Tensor> outputs = session.run(arena, List.of(inputTensor));
					var output = outputs.getFirst();

					float[] rawScores = output.data().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
					float[] probs = softmax(rawScores);

					IO.println("\nEmotion probabilities:");
					for (int i = 0; i < probs.length; i++) {
						IO.print("%-10s : %.2f%%%n".formatted(EMOTIONS[i], probs[i] * 100));
					}

					int pred = argMax(probs);
					IO.println("Predicted emotion: " + EMOTIONS[pred] + "\n");

				} catch (IOException e) {
					IO.println("Error reading image: " + e.getMessage());
				}
			}
		}
	}

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
			out[i] = (float) (exps[i] / sum);
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
}