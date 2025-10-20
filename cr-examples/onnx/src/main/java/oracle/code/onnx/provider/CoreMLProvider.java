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

package oracle.code.onnx.provider;

import oracle.code.onnx.OnnxRuntime;

import java.util.logging.Logger;
import java.util.logging.Level;

import static oracle.code.onnx.foreign.coreml_provider_factory_h.*;

public final class CoreMLProvider implements OnnxProvider {
    private static final Logger logger = Logger.getLogger(CoreMLProvider.class.getName());

    private int flag;

    public CoreMLProvider(int flag) {
        this.flag = flag;
    }

    @Override
    public void configure(OnnxRuntime.SessionOptions sessionOptions) {
        var sessionOptionsAddress = sessionOptions.getSessionOptionsAddress();

        try {
            var status = OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptionsAddress, flag);

            if (status == null || status.address() == 0) {
                logger.info("CoreML execution provider enabled successfully!");
            } else {
                logger.warning("CoreML EP returned status: " + status.address());

                status = OrtSessionOptionsAppendExecutionProvider_CoreML(
                        sessionOptionsAddress,
                        COREML_FLAG_USE_CPU_ONLY());
                if (status == null || status.address() == 0) {
                    logger.info("CoreML execution provider enabled with CPU_ONLY fallback!");
                } else {
                    logger.severe("CoreML EP failed with all flags - " + status.address());
                }
            }
        } catch (UnsatisfiedLinkError e) {
            logger.severe("CoreML execution provider is not available in the native ONNX Runtime library");
            throw new RuntimeException("CoreML execution provider is not available in the native ONNX Runtime library (symbol missing).", e);
        } catch (Throwable t) {
            logger.log(Level.SEVERE, "Unexpected error while enabling CoreML EP: " + t.getMessage(), t);
            throw new RuntimeException("Unexpected error while enabling CoreML EP: " + t.getMessage(), t);
        }
    }
}
