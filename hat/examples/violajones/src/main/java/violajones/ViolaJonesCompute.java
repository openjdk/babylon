/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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
package violajones;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.Schema;
import hat.backend.Backend;
import hat.backend.JavaMultiThreadedBackend;
import hat.backend.JavaSequentialBackend;
import hat.buffer.Buffer;
import hat.buffer.F32Array2D;
import org.xml.sax.SAXException;
import violajones.attic.ViolaJones;
import violajones.attic.ViolaJonesRaw;
import violajones.buffers.RgbS08x3Image;
import violajones.ifaces.Cascade;
import violajones.ifaces.ResultTable;
import violajones.ifaces.ScaleTable;

import javax.imageio.ImageIO;
import javax.xml.parsers.ParserConfigurationException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.runtime.CodeReflection;

public class ViolaJonesCompute {

    public static void main(String[] args) throws IOException, ParserConfigurationException, SAXException {
        boolean headless = Boolean.getBoolean("headless") ||( args.length>0 && args[0].equals("--headless"));

        String imageName = (args.length>2 && args[1].equals("--image"))?args[2]:System.getProperty("image", "Nasa1996");
        System.out.println("Using image "+imageName+".jpg");
        BufferedImage nasa1996 = ImageIO.read(ViolaJones.class.getResourceAsStream("/images/"+imageName+".jpg"));
               //"/images/team.jpg"
              // "/images/eggheads.jpg"
             // "/images/highett.jpg"
        //     "/images/Nasa1996.jpg"
      //  ));
        XMLHaarCascadeModel haarCascade = XMLHaarCascadeModel.load(
                ViolaJonesRaw.class.getResourceAsStream("/cascades/haarcascade_frontalface_default.xml"));
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(),
              //  new JavaSequentialBackend()
                Backend.FIRST
        );
        Cascade cascade = Cascade.create(accelerator, haarCascade);

       var cascade2 = Cascade.schema.allocate(accelerator,haarCascade.featureCount(),haarCascade.stageCount(),haarCascade.treeCount());


       System.out.println("Original   "+Buffer.getLayout(cascade));
        System.out.println("Schema     "+Buffer.getLayout(cascade2));
        RgbS08x3Image rgbImage = RgbS08x3Image.create(accelerator, nasa1996);
        ResultTable resultTable = ResultTable.create(accelerator, 1000);
        System.out.println("result table layout "+Buffer.getLayout(resultTable));
        HaarViewer harViz = null;
        if (!headless){
            harViz = new HaarViewer(accelerator, nasa1996, rgbImage, cascade, null, null);
        }

        //System.out.println("Compute units "+((NativeBackend)accelerator.backend).getGetMaxComputeUnits());
        for (int i = 0; i < 10; i++) {
            resultTable.atomicResultTableCount(0);
            accelerator.compute(cc ->
                    ViolaJonesCoreCompute.compute(cc, cascade, nasa1996, rgbImage, resultTable)
            );
            System.out.println(resultTable.atomicResultTableCount()+ "faces found");
        }
        if (harViz != null) {
            harViz.showResults(resultTable, null, null);
        }
    }
}
