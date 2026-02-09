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
package hat.util.ui;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JMenuBar;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSlider;
import javax.swing.JToggleButton;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public record Menu(JMenuBar menuBar) {
   public Menu exit() {
        ((JButton) menuBar.add(new JButton("Exit"))).addActionListener(_ -> System.exit(0));
        return this;
    }

    public  Menu space(int width) {
        menuBar.add(Box.createHorizontalStrut(width));
        return this;
    }

    public  Menu label(String text) {
        menuBar.add(new JLabel(text));
        return this;
    }

    public Menu toggle(String selected,String unselected, boolean sel, Consumer<JToggleButton> jToggleButtonConsumer,Consumer<Boolean> stateConsumer){
        JToggleButton toggleButton = (JToggleButton) menuBar.add(new JToggleButton(unselected));
        toggleButton.setSelected(sel);
        jToggleButtonConsumer.accept(toggleButton);
        toggleButton.addChangeListener(e -> {
                    stateConsumer.accept(toggleButton.isSelected());
                    toggleButton.setText(toggleButton.isSelected()?selected:unselected);
                }
        );
        return this;
    }
    public Menu toggle(String selected,String unselected, boolean sel, Consumer<Boolean> stateConsumer){
        return toggle(selected,unselected,sel,_->{}, stateConsumer);
    }

    public Menu slider(int min, int max, int value, Consumer<JSlider> sliderConsumer, Consumer<Integer> valueConsumer) {
        JSlider slider = (JSlider) menuBar.add(new JSlider(min, max));
        slider.setValue(value);
        sliderConsumer.accept(slider);
        slider.addChangeListener(e -> valueConsumer.accept(slider.getValue()));
        return this;
    }
    public Menu slider(int min, int max, int value,  Consumer<Integer> valueConsumer) {
      return slider(min,max,value,_->{},valueConsumer);
    }

    public <T> Menu combo(List<T> list, T selected, Consumer<JComboBox<T>> comboBoxConsumer, Consumer<T> valueConsumer) {
        JComboBox<T> comboBox = (JComboBox<T>) menuBar.add(new JComboBox<>(list.toArray()));
        comboBox.setSelectedIndex(list.indexOf(selected));
        comboBoxConsumer.accept(comboBox);
        comboBox.addActionListener(e -> valueConsumer.accept((T) comboBox.getSelectedItem()));
        return this;
    }
    public <T> Menu combo(List<T> list, T selected, Consumer<T> valueConsumer) {
       return combo(list,selected,_->{},valueConsumer);
    }

    public record pair<T>(T value, JRadioButton jRadioButton) {
    }

    private <T> Menu radio(int axis, List<T> list, Consumer<ButtonGroup> buttonGroupConsumer, Consumer<pair<T>> valueConsumer) {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, axis));
        var pairs = new ArrayList<pair<T>>();
        var buttonGroup = new ButtonGroup();
        list.forEach(t -> {
            var pair = new pair<T>(t, new JRadioButton(t.toString()));
            panel.add(pair.jRadioButton);
            buttonGroup.add(pair.jRadioButton);
            pair.jRadioButton.addActionListener(e -> valueConsumer.accept(pair));
            pairs.add(pair);
        });
        menuBar.add(panel);
        buttonGroupConsumer.accept(buttonGroup);
        return this;
    }

    public<T> Menu vradio(List<T> list, Consumer<ButtonGroup> buttonGroupConsumer, Consumer<pair<T>> valueConsumer) {
        return radio(BoxLayout.Y_AXIS, list, buttonGroupConsumer, valueConsumer);
    }
    public <T> Menu vradio(List<T> list,  Consumer<pair<T>> valueConsumer) {
        return vradio( list, _->{}, valueConsumer);
    }

    public <T> Menu hradio(List<T> list, Consumer<ButtonGroup> buttonGroupConsumer, Consumer<pair<T>> valueConsumer) {
        return radio(BoxLayout.X_AXIS, list, buttonGroupConsumer, valueConsumer);
    }
    public <T> Menu hradio(List<T> list,  Consumer<pair<T>> valueConsumer) {
        return hradio( list, _->{}, valueConsumer);
    }

    public Menu sevenSegment(int digits, int digitWidth, Consumer<SevenSegmentDisplay> sliderConsumer) {
        sliderConsumer.accept((SevenSegmentDisplay) menuBar.add(new SevenSegmentDisplay(digits, digitWidth, menuBar.getForeground(), menuBar.getBackground())));
        return this;
    }
}
