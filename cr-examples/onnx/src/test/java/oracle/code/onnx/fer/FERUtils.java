package oracle.code.onnx.fer;

import java.util.stream.IntStream;

/**
 * Utility methods for FER (Facial Expression Recognition) operations.
 */
public class FERUtils {
    
    /**
     * Format the top K emotion probabilities as HTML for display.
     * @param probs Array of emotion probabilities
     * @param k Number of top emotions to show
     * @param emotions Array of emotion names
     * @return HTML formatted string
     */
    public static String formatTopK(float[] probs, int k, String[] emotions) {
        int[] idxs = topK(probs, k);
        StringBuilder sb = new StringBuilder();
        for (int idx : idxs) {
            sb.append(String.format("%s : %.1f%%<br>", emotions[idx], probs[idx] * 100));
        }
        return sb.toString();
    }
    
    /**
     * Find the indices of the top K elements in an array.
     * @param arr Array to search
     * @param k Number of top elements to find
     * @return Array of indices sorted by value (descending)
     */
    public static int[] topK(float[] arr, int k) {
        return IntStream.range(0, arr.length)
                .boxed()
                .sorted((i, j) -> Float.compare(arr[j], arr[i]))
                .limit(k)
                .mapToInt(i -> i)
                .toArray();
    }
    
    /**
     * Find the index of the maximum value in an array.
     * @param arr Array to search
     * @return Index of maximum value
     */
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
    
    /**
     * Apply softmax to convert raw scores to probabilities.
     * @param scores Raw scores array
     * @return Probabilities array (sums to 1.0)
     */
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
