package oracle.code.onnx.bert;

import oracle.code.onnx.Tensor;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class BertTokenizer {
    record Batch(Tensor<Long> inputIds, Tensor<Long> attentionMask, Tensor<Long> tokenTypeIds) {}

    private static final Long CLS_TOKEN_ID = 101L;
    private static final int MAX_TOKENS = 512; // max_position_embeddings
    private static final long SEP_TOKEN_ID = 102L;
    private static final long UNK_TOKEN_ID = 100L;

    private static final Set<Integer> PUNCTUATION_CATEGORIES = Set.of((int)Character.CONNECTOR_PUNCTUATION,
            (int)Character.DASH_PUNCTUATION,
            (int)Character.START_PUNCTUATION,
            (int)Character.END_PUNCTUATION,
            (int)Character.INITIAL_QUOTE_PUNCTUATION,
            (int)Character.FINAL_QUOTE_PUNCTUATION,
            (int)Character.OTHER_PUNCTUATION);

    Map<String, Long> vocab;

    public BertTokenizer() throws IOException, URISyntaxException {
        Path vocabLocation = Path.of(Objects.requireNonNull(BertTokenizer.class.getResource("vocab.txt")).toURI());
        List<String> lines = Files.readAllLines(vocabLocation);
        vocab = new HashMap<>(lines.size() * 2);
        for (int i = 0; i < lines.size(); i++) {
            vocab.put(lines.get(i), (long) i);
        }
    }

    Batch tokenize(Arena arena, String[] sentences) {
        List<long[]> tokenized = new ArrayList<>(sentences.length);
        int sequenceLength = 0;
        for (String sentence : sentences) {
            long[] ids = encode(sentence);
            tokenized.add(ids);
            sequenceLength = Math.max(sequenceLength, ids.length);
        }

        long[] shape = {sentences.length, sequenceLength};
        long[] inputIds = new long[sentences.length * sequenceLength];
        long[] attentionMask = new long[inputIds.length];
        long[] tokenTypeIds = new long[inputIds.length];
        for (int row = 0; row < tokenized.size(); row++) {
            long[] ids = tokenized.get(row);
            int offset = row * sequenceLength;
            System.arraycopy(ids, 0, inputIds, offset, ids.length);
            Arrays.fill(attentionMask, offset, offset + ids.length, 1L);
        }

        return new Batch(
                Tensor.ofShape(arena, shape, inputIds),
                Tensor.ofShape(arena, shape, attentionMask),
                Tensor.ofShape(arena, shape, tokenTypeIds));
    }

    private long[] encode(String sentence) {
        List<Long> ids = new ArrayList<>();
        ids.add(CLS_TOKEN_ID);
        for (String token : basicTokens(sentence)) {
            for (long id : wordPiece(token)) {
                if (ids.size() == MAX_TOKENS - 1) {
                    ids.add(SEP_TOKEN_ID);
                    return ids.stream().mapToLong(Long::longValue).toArray();
                }
                ids.add(id);
            }
        }
        ids.add(SEP_TOKEN_ID);
        return ids.stream().mapToLong(Long::longValue).toArray();
    }

    private static List<String> basicTokens(String sentence) {
        String normalized = sentence.toLowerCase(Locale.ENGLISH);
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < normalized.length(); i++) {
            char c = normalized.charAt(i);
            if (Character.isWhitespace(c) || Character.isISOControl(c)) {
                flush(current, tokens);
            } else if (PUNCTUATION_CATEGORIES.contains(Character.getType(c))) {
                flush(current, tokens);
                tokens.add(String.valueOf(c));
            } else {
                current.append(c);
            }
        }
        flush(current, tokens);
        return tokens;
    }

    private static void flush(StringBuilder current, List<String> tokens) {
        if (!current.isEmpty()) {
            tokens.add(current.toString());
            current.setLength(0);
        }
    }

    private long[] wordPiece(String token) {
        Long direct = vocab.get(token);
        if (direct != null) {
            return new long[] {direct};
        }

        List<Long> pieces = new ArrayList<>();
        int start = 0;
        while (start < token.length()) {
            int end = token.length();
            Long piece = null;
            while (start < end) {
                String candidate = start == 0 ? token.substring(start, end) : "##" + token.substring(start, end);
                piece = vocab.get(candidate);
                if (piece != null) {
                    break;
                }
                end--;
            }
            if (piece == null) {
                return new long[] {UNK_TOKEN_ID};
            }
            pieces.add(piece);
            start = end;
        }
        return pieces.stream().mapToLong(Long::longValue).toArray();
    }
}
