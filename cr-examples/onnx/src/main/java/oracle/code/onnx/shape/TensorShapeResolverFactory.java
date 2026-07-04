/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class TensorShapeResolverFactory {

    private TensorShapeResolverFactory() {
    }

    public static TensorShapeResolver fromConfig(Path configPath) throws IOException {
        String config = Files.readString(configPath);
        if (configPath.endsWith("genai_config.json")) {
            return new LLMTensorShapeResolver(
                    longValue(config, "vocab_size"),
                    longValue(config, "num_key_value_heads"),
                    longValue(config, "head_size"),
                    stringValue(config, "input_ids"),
                    stringValue(config, "attention_mask"),
                    stringValue(config, "logits"),
                    stringValue(config, "past_key_names"),
                    stringValue(config, "past_value_names"),
                    stringValue(config, "present_key_names"),
                    stringValue(config, "present_value_names"));
        }
        throw new UnsupportedOperationException("Language model configuration not handled " + configPath);
    }

    private static long longValue(String json, String key) throws IOException {
        int start = valueStart(json, key);
        int end = start;

        while (end < json.length() && Character.isDigit(json.charAt(end))) {
            end++;
        }

        if (end > start)
            return Long.parseLong(json.substring(start, end));
        throw new IOException("Missing numeric GenAI config field " + key);
    }

    private static String stringValue(String json, String key) throws IOException {
        int start = valueStart(json, key);

        if(start < json.length() && json.charAt(start) == 34) {
            int end = json.indexOf(34, start + 1);
            if (end > start)
                return json.substring(start + 1, end);
        }
        throw new IOException("Missing numeric GenAI config field " + key);
    }

    private static int valueStart(String json, String key) throws IOException {
        String keyToken = "\"" + key + "\"";
        int keyIndex = json.indexOf(keyToken);

        if (keyIndex < 0)
            throw new IOException("Missing GenAI config field" + key);

        int colon = json.indexOf(58, keyIndex + keyToken.length());
        if (colon < 0) {
            throw new IOException("Missing value for GenAI config field " + key);
        }

        int start = colon + 1;
        while (start < json.length() && Character.isWhitespace(json.charAt(start)))
            start++;
        return start;
    }
}
