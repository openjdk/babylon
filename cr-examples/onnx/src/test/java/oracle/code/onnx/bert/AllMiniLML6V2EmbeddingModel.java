package oracle.code.onnx.bert;

import jdk.incubator.code.Reflect;
import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.genai.TensorDataStream;

import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.net.URISyntaxException;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.util.*;
import java.util.function.Supplier;

import static java.util.Optional.*;
import static oracle.code.onnx.OnnxOperators.*;
import static oracle.code.onnx.Tensor.ElementType.FLOAT;

public class AllMiniLML6V2EmbeddingModel {
    public static final int LAYERS = 6;

    public static final long HIDDEN_SIZE = 384, // hidden_size
            INTERMEDIATE_SIZE = 1536, // intermediate_size
            NUM_HEADS = 12, // num_attention_heads
            MAX_POSITION_EMBEDDINGS = 512, // max_position_embeddings
            TYPE_VOCAB_SIZE = 2, // type_vocab_size
            VOCAB_SIZE = 30522; // vocab_size
    public static final float EPSILON = 1.0E-12f, // layer_norm_eps
            ATTENTION_SCALE = 0.17677669f; // 1/sqrt(HIDDEN_SIZE/NUM_HEADS) = 1/ sqrt(32)

    public final Tensor<Long> scalar0, scalar1, axis0, axis1, axis2;
    public final Tensor<Float> wordEmbeddingWeight, tokenTypeEmbeddingWeight, positionEmbeddingWeight, embeddingNormWeight, embeddingNormBias;
    public final Tensor<Float>[] attnQWeight = new Tensor[LAYERS],
            attnKWeight = new Tensor[LAYERS],
            attnVWeight = new Tensor[LAYERS],
            attnOWeight = new Tensor[LAYERS],
            attnQBias = new Tensor[LAYERS],
            attnKBias = new Tensor[LAYERS],
            attnVBias = new Tensor[LAYERS],
            attnOBias = new Tensor[LAYERS],
            attentionOutputNormWeight = new Tensor[LAYERS],
            attentionOutputNormBias = new Tensor[LAYERS],
            mlpFc1Weight = new Tensor[LAYERS],
            mlpFc1Bias = new Tensor[LAYERS],
            mlpFc2Weight = new Tensor[LAYERS],
            mlpFc2Bias = new Tensor[LAYERS],
            mlpOutputNormWeight = new Tensor[LAYERS],
            mlpOutputNormBias = new Tensor[LAYERS];

    private final BertTokenizer tokenizer;

    public AllMiniLML6V2EmbeddingModel(Arena arena) throws IOException, URISyntaxException {
        scalar0 = Tensor.ofScalar(arena, 0L);
        scalar1 = Tensor.ofScalar(arena, 1L);
        axis0 = Tensor.ofFlat(arena, 0L);
        axis1 = Tensor.ofFlat(arena, 1L);
        axis2 = Tensor.ofFlat(arena, 2L);
        tokenizer = new BertTokenizer();

        var modelData = new TensorDataStream(arena, Objects.requireNonNull(AllMiniLML6V2EmbeddingModel.class.getResource("model.onnx_data")).getPath());
        wordEmbeddingWeight = modelData.nextTensor(FLOAT, VOCAB_SIZE, HIDDEN_SIZE);
        tokenTypeEmbeddingWeight = modelData.nextTensor(FLOAT, TYPE_VOCAB_SIZE, HIDDEN_SIZE);
        positionEmbeddingWeight = modelData.nextTensor(FLOAT, MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE);
        embeddingNormWeight = modelData.nextTensor(FLOAT, HIDDEN_SIZE);
        embeddingNormBias = modelData.nextTensor(FLOAT, HIDDEN_SIZE);

        for (int i = 0; i < LAYERS; i++) {
            attnQWeight[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE, HIDDEN_SIZE);
            attnKWeight[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE, HIDDEN_SIZE);
            attnVWeight[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE, HIDDEN_SIZE);
            attnQBias[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE);
            attnKBias[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE);
            attnVBias[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE);
            attnOWeight[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE, HIDDEN_SIZE);
            attnOBias[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE);
            attentionOutputNormWeight[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE);
            attentionOutputNormBias[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE);
            mlpFc1Weight[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE, INTERMEDIATE_SIZE);
            mlpFc1Bias[i] = modelData.nextTensor(FLOAT, INTERMEDIATE_SIZE);
            mlpFc2Weight[i] = modelData.nextTensor(FLOAT, INTERMEDIATE_SIZE, HIDDEN_SIZE);
            mlpFc2Bias[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE);
            mlpOutputNormWeight[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE);
            mlpOutputNormBias[i] = modelData.nextTensor(FLOAT, HIDDEN_SIZE);
        }
    }

    public Embedding embed(Arena arena, String... sentences) {
        BertTokenizer.Batch batch = tokenizer.tokenize(arena, sentences);
        Tensor<Long> inputIds = batch.inputIds();
        Tensor<Long> attentionMask = batch.attentionMask();
        Tensor<Long> tokenTypeIds = batch.tokenTypeIds();

        ForwardResponse response = OnnxRuntime.execute(arena,
                MethodHandles.lookup(),
                (@Reflect Supplier<ForwardResponse>) () -> forward(inputIds, attentionMask, tokenTypeIds)
        );
        return parse(response.embeddings);
    }

    private static Embedding parse(Tensor<Float> tensor) {
        long[] shape = tensor.shape();
        if (shape.length != 2) {
            throw new IllegalStateException("Expected embeddings with shape [batch, width], got " + Arrays.toString(shape));
        }
        int rows = Math.toIntExact(shape[0]);
        int columns = Math.toIntExact(shape[1]);
        long elements = Math.multiplyExact(shape[0], shape[1]);
        float[] data = tensor.data()
                .reinterpret(elements * ValueLayout.JAVA_FLOAT.byteSize())
                .toArray(ValueLayout.JAVA_FLOAT);
        return new Embedding(rows, columns, data);
    }

    public record Embedding(int rows, int columns, float[] data) {
        public float[] row(int row) {
            if (row < 0 || row >= rows) {
                throw new IndexOutOfBoundsException(row);
            }
            return Arrays.copyOfRange(data, row * columns, (row + 1) * columns);
        }
    }

    public record ForwardResponse(Tensor<Float> embeddings) {
    }

    @Reflect
    public ForwardResponse forward(Tensor<Long> inputIds, Tensor<Long> attentionMask, Tensor<Long> tokenTypeIds) {
        Tensor<Long> inputShape = Shape(inputIds, empty(), empty());
        Tensor<Long> sequenceLength = Gather(inputShape, scalar1, of(0L));
        Tensor<Long> positions = Unsqueeze(Range(scalar0, sequenceLength, scalar1, empty()), axis0);

        Tensor<Float> wordEmbeddings = Gather(wordEmbeddingWeight, inputIds, empty());
        Tensor<Float> tokenTypeEmbeddings = Gather(tokenTypeEmbeddingWeight, tokenTypeIds, empty());
        Tensor<Float> positionEmbeddings = Gather(positionEmbeddingWeight, positions, empty());
        Tensor<Float> hidden = LayerNormalization(
                Add(Add(wordEmbeddings, positionEmbeddings), tokenTypeEmbeddings),
                embeddingNormWeight,
                of(embeddingNormBias),
                of(EPSILON),
                of(1L),
                of(-1L)).Y();

        Tensor<Float> mask = Cast(attentionMask, empty(), 1L, empty());
        Tensor<Integer> keyPaddingMask = Cast(attentionMask, empty(), 6L, empty());

        for (int layer = 0; layer < LAYERS; layer++) {
            Tensor<Float> query = Add(MatMul(hidden, attnQWeight[layer]), attnQBias[layer]);
            Tensor<Float> key = Add(MatMul(hidden, attnKWeight[layer]), attnKBias[layer]);
            Tensor<Float> value = Add(MatMul(hidden, attnVWeight[layer]), attnVBias[layer]);
            var attention = MultiHeadAttention(
                    query,
                    key,
                    value,
                    java.util.Optional.<Tensor<Float>>empty(),
                    of(keyPaddingMask),
                    java.util.Optional.<Tensor<Float>>empty(),
                    java.util.Optional.<Tensor<Float>>empty(),
                    java.util.Optional.<Tensor<Float>>empty(),
                    java.util.Optional.<Tensor<Float>>empty(),
                    NUM_HEADS,
                    java.util.Optional.<Float>empty(),
                    of(ATTENTION_SCALE),
                    java.util.Optional.<Long>empty());

            Tensor<Float> attentionResidual = Add(attention.output(), hidden);
            Tensor<Float> attentionNorm = LayerNormalization(
                    attentionResidual,
                    attentionOutputNormWeight[layer],
                    of(attentionOutputNormBias[layer]),
                    of(EPSILON),
                    of(1L),
                    of(-1L)).Y();

            Tensor<Float> fc1 = Add(MatMul(attentionNorm, mlpFc1Weight[layer]), mlpFc1Bias[layer]);
            Tensor<Float> gelu = Gelu(fc1, empty());
            Tensor<Float> fc2 = Add(MatMul(gelu, mlpFc2Weight[layer]), mlpFc2Bias[layer]);
            Tensor<Float> mlpResidual = Add(fc2, attentionNorm);
            hidden = LayerNormalization(
                    mlpResidual,
                    mlpOutputNormWeight[layer],
                    of(mlpOutputNormBias[layer]),
                    of(EPSILON),
                    of(1L),
                    of(-1L)).Y();
        }

        Tensor<Float> expandedMask = Unsqueeze(mask, axis2);
        Tensor<Float> embeddingSums = ReduceSum(Mul(hidden, expandedMask), of(axis1), empty(), of(0L));
        Tensor<Float> tokenCounts = ReduceSum(mask, of(axis1), empty(), of(0L));
        Tensor<Float> pooled = Div(embeddingSums, Unsqueeze(tokenCounts, axis1));
        Tensor<Float> normSquares = ReduceSum(Mul(pooled, pooled), of(axis1), empty(), of(1L));
        Tensor<Float> embeddings = Div(pooled, Sqrt(normSquares));
        return new ForwardResponse(embeddings);
    }
}