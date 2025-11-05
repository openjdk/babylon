/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.onnx.fer;

import oracle.code.onnx.OnnxProvider;
import oracle.code.onnx.OnnxRuntime;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URL;
import java.util.Objects;

import static oracle.code.onnx.fer.FERCoreMLDemo.IMAGE_SIZE;

public class FERInference {


    private final OnnxRuntime runtime;

	public FERInference() {
        runtime = OnnxRuntime.getInstance();
    }

    public float[] analyzeImage(Arena arena, OnnxRuntime.SessionOptions sessionOptions, URL url, boolean isCondensed) throws Exception {
        float[] imageData = transformToFloatArray(url);
		FERModel ferModel = new FERModel(arena);
        float[] rawScores = ferModel.classify(imageData, sessionOptions, isCondensed);
        return rawScores;
    }

	public OnnxRuntime.SessionOptions prepareSessionOptions(Arena arena, OnnxProvider provider) {
		var sessionOptions = runtime.createSessionOptions(arena);
		if (Objects.nonNull(provider)) {
			runtime.appendExecutionProvider(arena, sessionOptions, provider);
		}
		return sessionOptions;
	}

	private float[] transformToFloatArray(URL imgUrl) throws IOException {
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
