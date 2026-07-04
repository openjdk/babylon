/*
 * Copyright (c) 2026 Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package oracle.code.onnx.shape;

import oracle.code.onnx.llm.LlamaDemo;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static oracle.code.onnx.shape.TensorShapeResolverFactory.GENAI_CONFIG_JSON;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumingThat;

class TensorShapeResolverFactoryTest {

    @Test
    void fromConfigWithValidContent() {
        assumingThat(Files.exists(Path.of(Objects.requireNonNull(LlamaDemo.class.getResource(GENAI_CONFIG_JSON)).getPath())),
                () -> {
                    Path genConfigLocation = Path.of(Objects.requireNonNull(LlamaDemo.class.getResource(GENAI_CONFIG_JSON)).getPath());
                    var resolver = TensorShapeResolverFactory.fromConfig(genConfigLocation);
                    assertEquals(List.of("batch_size", "sequence_length"), resolver.shape("inputIds"));
                    assertEquals(List.of("batch_size", "sequence_length"), resolver.shape("attentionMask"));
                    assertEquals(List.of("batch_size", "sequence_length", 128256L), resolver.shape("logits"));
                    assertEquals(List.of("batch_size", 8L, "past_sequence_length", 64L), resolver.shape("pastKey.0"));
                    assertEquals(List.of("batch_size", 8L, "past_sequence_length", 64L), resolver.shape("pastValue.15"));
                    assertEquals(List.of("batch_size", 8L, "total_sequence_length", 64L), resolver.shape("presentKey.0"));
                    assertEquals(List.of("batch_size", 8L, "total_sequence_length", 64L), resolver.shape("presentValue.15"));
                    assertNull(resolver.shape("unknown"));
                });
    }

    @Test
    void fromConfigWithWrongContent(@TempDir Path tempDir) throws IOException {
        String fixture = """
                {
                    "model": {
                        "decoder": {
                            "filename": "model.onnx",
                            "head_size": 64,
                            "hidden_size": 2048,
                            "inputs": {
                                "input_ids": "input_ids",
                                "attention_mask": "attention_mask",
                                "past_key_names": "pastKey.%d",
                                "past_value_names": "pastValue.%d"
                            },
                            "outputs": {
                                "logits": "logits",
                                "present_key_names": "presentKey.%d",
                                "present_value_names": "presentValue.%d"
                            },
                            "num_attention_heads": 32,
                            "num_hidden_layers": 16,
                            "num_key_value_heads": 8
                        }
                        "pad_token_id": 128001,
                        "type": "llama",
                        "vocab_size": 128256
                    }
                }
                """;
        Path path = Path.of(GENAI_CONFIG_JSON);
        Files.writeString(path, fixture, StandardCharsets.UTF_8);

        var resolver = TensorShapeResolverFactory.fromConfig(path);
        assertNull(resolver.shape("inputIds"));
        assertNull(resolver.shape("attentionMask"));
        assertEquals(List.of("batch_size", "sequence_length", 128256L), resolver.shape("logits"));
        assertEquals(List.of("batch_size", 8L, "past_sequence_length", 64L), resolver.shape("pastKey.0"));
        assertEquals(List.of("batch_size", 8L, "past_sequence_length", 64L), resolver.shape("pastValue.15"));
        assertEquals(List.of("batch_size", 8L, "total_sequence_length", 64L), resolver.shape("presentKey.0"));
        assertEquals(List.of("batch_size", 8L, "total_sequence_length", 64L), resolver.shape("presentValue.15"));
        assertNull(resolver.shape("unknown"));
    }

    @Test
    void fromConfigThrowsException() throws IOException {
        Path path = Path.of("config.json");
        String fixture = "";
        Files.writeString(path, fixture, StandardCharsets.UTF_8);
        assertThrows(UnsupportedOperationException.class, () -> TensorShapeResolverFactory.fromConfig(path));
    }
}