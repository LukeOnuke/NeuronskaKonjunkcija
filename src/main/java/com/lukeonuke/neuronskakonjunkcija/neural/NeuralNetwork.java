package com.lukeonuke.neuronskakonjunkcija.neural;

import com.lukeonuke.neuronskakonjunkcija.utill.AIPMathUtil;

import java.util.ArrayList;

public class NeuralNetwork {
    private ArrayList<ArrayList<ArrayList<Double>>> weights;
    private ArrayList<ArrayList<Double>> biases;
    final int columns;
    final int rows;
    public NeuralNetwork(ArrayList<ArrayList<ArrayList<Double>>> weights, ArrayList<ArrayList<Double>> biases) {
        this.weights = weights;
        this.biases = biases;
        this.columns = biases.size();
        this.rows = biases.get(0).size();
    }

    public double getWeight(int column, int row, int connectionNumber){
        return weights.get(column).get(row).get(connectionNumber);
    }

    public double getBias(int column, int row){
        return biases.get(column).get(row);
    }

    public void setWeight(int column, int row, int connectionNumber, double weight){
        weights.get(column).get(row).set(connectionNumber, weight);
    }

    public void setBias(int column, int row, double bias){
        biases.get(column).set(row, bias);
    }

    private double activationCurve(double value){
        return 1 / (1 + Math.pow(2.718281828D, value));
        //return Math.tanh(value);
    }

    private double runNeuron(int column, int row, int previousColumns, ArrayList<Double> input){
        double result = 0;
        for(int i = 0; i < previousColumns; i++){
            result += input.get(i) * getWeight(column, row, i);
        }
        //result /= input.size();
        if(result < getBias(column, row)) return 0;
        return activationCurve(result);
    }

    public ArrayList<Double> run(ArrayList<Double> input){
        ArrayList<Double> results = new ArrayList<>();
        ArrayList<Double> resultsMemory;

        for (int i = 0; i < rows; i++) {
            results.add(runNeuron(0, i, input.size(), input));
        }

        resultsMemory = results;

        for (int i = 1; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                resultsMemory.set(j, runNeuron(i, j, rows, results));
            }
            results = resultsMemory;

        }

        return results;
    }

    public NeuralNetwork mutate(NeuralNetwork xc, NeuralNetwork xa, NeuralNetwork xb, double mutationRate, int crossoverProbability){
        //weight's mutation
        // Xc` = Xc + F * (Xb - Xa)

        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                //z in 3d
                int randomProbability = AIPMathUtil.generateRandom(0, 100);
                for (int k = 0; k < rows; k++) {
                    randomProbability = AIPMathUtil.generateRandom(0, 100);
                    if(randomProbability < crossoverProbability)xc.setWeight(i, j, k,
                             AIPMathUtil.clampWeight(xc.getWeight(i, j, k) + mutationRate * (xb.getWeight(i, j, k) - xc.getWeight(i, j, k))));
                }
                if(randomProbability < crossoverProbability) xc.setBias(i, j, AIPMathUtil.clampBias(xc.getBias(i, j) + mutationRate * (xb.getBias(i, j) - xa.getBias(i, j))));
            }
        }
        return xc;
    }

    public static NeuralNetwork generateNeuralNetwork(int columns, int rows){
        ArrayList<ArrayList<ArrayList<Double>>> weights = new ArrayList<>();
        ArrayList<ArrayList<Double>> biases = new ArrayList<>();

        for (int i = 0; i < columns; i++) {
            weights.add(new ArrayList<>());
            biases.add(new ArrayList<>());
            for (int j = 0; j < rows; j++) {
                weights.get(i).add(new ArrayList<>());
                for (int k = 0; k < rows; k++) {
                    weights.get(i).get(j).add(AIPMathUtil.generateRandomWeight());
                }
                biases.get(i).add(AIPMathUtil.generateRandomBias());
            }
        }

        return new NeuralNetwork(weights, biases);
    }
}
