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
package logger;
import org.testng.ITestResult;
import org.testng.TestListenerAdapter;

public class HATTestNGLogger  extends TestListenerAdapter{
        private int m_count = 0;
        @Override
        public void onTestFailure(ITestResult tr) {
            log(tr.getName()+ "--Test method failed\n");
        }

        @Override
        public void onTestSkipped(ITestResult tr) {
            log(tr.getName()+ "--Test method skipped\n");
        }

        @Override
        public void onTestSuccess(ITestResult tr) {
            log(tr.getName()+ "--Test method success\n");
        }

        private void log(String string) {
            System.out.print(string);
           // if (++m_count % 40 == 0) {
                System.out.println("");
           // }
        }


}
