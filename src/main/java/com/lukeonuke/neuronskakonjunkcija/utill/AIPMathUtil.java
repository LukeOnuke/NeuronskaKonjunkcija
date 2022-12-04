package com.lukeonuke.neuronskakonjunkcija.utill;

import java.util.Random;

public class AIPMathUtil {
    public static double generateRandom(double min, double max){
        return new Random().nextDouble(min, max);
    }

    public static int generateRandom(int min, int max){
        return new Random().nextInt(min, max);
    }

    public static double generateRandomWeight(){
        return generateRandom(-1D, 1D);
    }

    public static double generateRandomBias(){
        return generateRandom(-2, 2);
    }

    public static double clamp(double value, double min, double max){
        return Math.max(min, Math.min(max, value));
    }

    public static double clampBias(double bias){
        return clamp(bias, -2, 2D);
    }

    public static double clampWeight(double weight){
        return clamp(weight, -1D, 1D);
    }
}
