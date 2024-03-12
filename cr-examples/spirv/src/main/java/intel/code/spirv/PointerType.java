/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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

package intel.code.spirv;

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.TypeDefinition;
import java.util.Objects;
import java.util.List;

public final class PointerType extends SpirvType {
	static final String NAME = "spirv.pointer";
	private final TypeElement referentType;
	private final TypeElement storageType;

	public PointerType(TypeElement referentType, TypeElement storageType)
	{
		this.referentType = referentType;
		this.storageType = storageType;
	}

	public TypeElement referentType()
	{
		return referentType;
	}

	public TypeElement storageType()
	{
		return storageType;
	}

	@Override
	public boolean equals(Object obj)
	{
		if (obj == null || obj.getClass() != PointerType.class) return false;
		PointerType pt = (PointerType)obj;
		return pt.referentType().equals(referentType) && pt.storageType.equals(storageType);
	}
	
	@Override
    public int hashCode() {
        return Objects.hash(referentType, storageType);
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        return new TypeDefinition(NAME, List.of(referentType.toTypeDefinition(), storageType.toTypeDefinition()));
    }

    @Override
    public String toString() {
        return toTypeDefinition().toString();
    }
}