package com.lukeonuke.neuronskakonjunkcija;

import com.formdev.flatlaf.FlatLightLaf;
import com.lukeonuke.neuronskakonjunkcija.gui.MainWindow;

public class Main {
    public static void main(String[] args) {
        System.out.println("IT RUNS");
        FlatLightLaf.setup();
        new MainWindow();
        //new NeuralNetworkHost(6000).runTraining();
    }
}
