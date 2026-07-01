package oracle.code.onnx.bert;

import java.lang.foreign.Arena;
import java.util.Arrays;

import oracle.code.onnx.bert.AllMiniLML6V2EmbeddingModel.Embedding;

public class EmbeddingDemo {

    public static void main(String[] args) throws Exception {
        String[] sentences = args.length == 0
                ? new String[]{"This is an example sentence", "Each sentence is converted"}
                : args;

        try (Arena arena = Arena.ofConfined()) {
            var modelInstance = new AllMiniLML6V2EmbeddingModel(arena);
            Embedding embedding = modelInstance.embed(arena, sentences);
            System.out.println("dims: [" + embedding.rows() + ", " + embedding.columns() + "]");
            for (int i = 0; i < embedding.rows(); i++) {
                float[] embeddings = embedding.row(i);
                System.out.println('"' + sentences[i] + '"');
                System.out.println(Arrays.toString(Arrays.copyOfRange(embeddings, 0, 12)) + " ...");
            }
        }
    }

}

