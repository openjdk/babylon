package oracle.code.onnx.fer;

import java.util.stream.IntStream;

public class FERUtils {
    
    public static String formatTopK(float[] probs, int k, String[] emotions) {
        int[] idxs = topK(probs, k);
        StringBuilder sb = new StringBuilder();
        for (int idx : idxs) {
            sb.append(String.format("%s : %.1f%%<br>", emotions[idx], probs[idx] * 100));
        }
        return sb.toString();
    }
    
    public static int[] topK(float[] arr, int k) {
        return IntStream.range(0, arr.length)
                .boxed()
                .sorted((i, j) -> Float.compare(arr[j], arr[i]))
                .limit(k)
                .mapToInt(i -> i)
                .toArray();
    }
    
    public static int argMax(float[] arr) {
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
    
    public static float[] softmax(float[] scores) {
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
}
