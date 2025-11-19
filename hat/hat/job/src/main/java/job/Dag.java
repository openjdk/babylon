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
package job;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Dag {
        record DotBuilder(Consumer<String> consumer){
           // https://graphviz.org/doc/info/lang.html
            static DotBuilder of(Consumer<String> stringConsumer, Consumer<DotBuilder> builderConsumer){
                DotBuilder db = new DotBuilder(stringConsumer);
                db.append("strict digraph graphname {").append("\n");
                db.append("   node [shape=record];\n");
                builderConsumer.accept(db);
                db.append("\n}");
                return db;
            }
            DotBuilder append(String s){
                consumer.accept(s);
                return this;
            }
            DotBuilder quoted(String s){
                append("\"").append(s).append("\"");
                return this;
            }
            DotBuilder node(String n, String label){
                return append("\n   ").quoted(n).append("[").append("label").append("=").quoted(label).append("]").append(";");
            }
            DotBuilder edge(String from, String to){
                return append("\n   ").quoted(from).append("->").quoted(to).append(";");
            }
        }
        Map<Dependency, Set<Dependency>> map = new LinkedHashMap<>();
        record Edge(Dependency from, Dependency to) {}
        List<Edge> edges = new ArrayList<>();
         public void recurse( Dependency from) {
            var set = map.computeIfAbsent(from, _ -> new LinkedHashSet<>());
            var deps = from.dependencies();
            deps.forEach(dep -> {
                edges.add(new Edge(from, dep));
                set.add(dep);
                recurse( dep);
            });
        }
        public Dag(Set<Dependency> deps) {
            deps.forEach(this::recurse);
        }
        public Dag(Dependency...deps) {
             this(Stream.of(deps).collect(Collectors.toSet()));
        }

        public String toDot(){
            StringBuilder sb = new StringBuilder();
            DotBuilder.of(sb::append, db-> {
                map.keySet().forEach(k -> {
                    db.node(k.id().projectRelativeHyphenatedName(), k.id().projectRelativeHyphenatedName());
                });
                edges.forEach(e ->
                        db.edge(e.from.id().projectRelativeHyphenatedName(), e.to.id().projectRelativeHyphenatedName())
                );
            });
            return sb.toString();
        }
        public Set<Dependency> ordered(){
            Set<Dependency> ordered = new LinkedHashSet<>();
            while (!map.isEmpty()) {
                var leaves = map.entrySet().stream()
                        .filter(e -> e.getValue().isEmpty())    // if this entry has zero dependencies
                        .map(Map.Entry::getKey)                 // get the key
                        .collect(Collectors.toSet());
                map.values().forEach(v -> leaves.forEach(v::remove));
                leaves.forEach(leaf -> {
                    map.remove(leaf);
                    ordered.add(leaf);
                });
            }
            return ordered;
        }

    public Dag available(){
        var ordered = this.ordered();
        Set<Dependency> unavailable = ordered.stream().filter(
                d -> {
                    if (d instanceof Dependency.Optional opt) {
                       return !opt.isAvailable();
                    }else{
                        return false;
                    }
                })
                .collect(Collectors.toSet());

        boolean changed = true;
        while (changed) {
            changed = false;
            for(Dependency dep : ordered) {
                if (!changed) {
                    var optionalDependsOnUnavailable = dep.dependencies().stream().filter(d ->
                            unavailable.contains(d) || d instanceof Dependency.Optional o && !o.isAvailable()).findFirst();
                    if (optionalDependsOnUnavailable.isPresent()) {
                        changed = true;
                        unavailable.add(dep);
                        ordered.remove(dep);
                        break;
                    }
                }
            }
        }
        return new Dag(ordered);
    }


}
