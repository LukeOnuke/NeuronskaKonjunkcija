package com.lukeonuke.neuronskakonjunkcija.neural.runtime;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.lukeonuke.neuronskakonjunkcija.neural.NeuralNetwork;
import com.lukeonuke.neuronskakonjunkcija.utill.AIPMathUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicReference;

public class NeuralNetworkHost {
    /*
    *   for(i = 0; i < NP; i++){
	*   izaberi cetri randomna iz pop xi, xa, xb, xc
	*   Xc` = Xc + F * (Xb - Xa) //F je tezina evolucije (0.5-1)
    *
	    for(veza : xi) {
		    xt = xi nasledi od Xc` sa verovatnocom CR
		    //mora makar jedno polje da se nasledi kako god
		    if(xt je bolji od xi) xi = xt;
	    }
    }
    * */
    Vector<NeuralNetwork> candidates = new Vector<>();

    final double mutationRate = 0.25;
    final int crossoverProbability = 90;

    public NeuralNetworkHost(int candidateAmount) {
        for (int i = 0; i < candidateAmount; i++) {
            candidates.add(NeuralNetwork.generateNeuralNetwork(4, 4));
        }
    }

    static Double[][] X = {
            {0D, 0D},
            {1D, 0D},
            {0D, 1D},
            {1D, 1D}
    };
    static Double[] Y = {
            0D, 1D, 1D, 1D
    };

    private double testWorthy(NeuralNetwork neuralNetwork, int datasetIndex) {
//        double maxError = Double.MIN_VALUE;
//        for (int i = 0; i < 3; i++) {
//            ArrayList<Double> results = neuralNetwork.run(new ArrayList<Double>(Arrays.asList(X[i])));
//            double e = Math.abs(Math.abs(results.get(0)) - Y[i]);
//            if(e > maxError) maxError = e;
//        }
////        return maxError;
//
//        ArrayList<Double> results = neuralNetwork.run(new ArrayList<Double>(Arrays.asList(X[datasetIndex])));
//        System.out.println(results);
//        return Math.abs(Math.abs(results.get(0)) - Y[datasetIndex]);

        //double score = 0;
        //for (int i = 0; i < 4; i++) {
            ArrayList<Double> results = neuralNetwork.run(new ArrayList<Double>(Arrays.asList(X[datasetIndex])));
            double diff = Math.abs(0.5 - results.get(datasetIndex)) * 1000; //1000

            if (results.get(0) <= 0.5) {
                // 0
                if (0 == Y[datasetIndex]) {
                    return  -50000 - diff; //-50000
                }else{
                    return diff;
                }
            } else {
                // 1
                if (1 == Y[datasetIndex]) {
                    return  -50000 - diff;
                }else{
                    return diff;
                }
            }
    }

    public void runTraining() {
        final int candidateAmount = candidates.size();
        int a, b, c;

        double minDelta = Double.MAX_VALUE;
        boolean train;
        long epoch = 0;

        //for(int j =0; j < 10000; j++){
        do {
            train = false;
            for (int i = 0; i < candidateAmount; i++) {
                int datasetIndex = AIPMathUtil.generateRandom(0, 4);

                //Izabrati nasumicno a, b, c pod uslovom a!=b!=c.
                do {
                    a = AIPMathUtil.generateRandom(0, candidateAmount);
                } while (a == i);
                do {
                    b = AIPMathUtil.generateRandom(0, candidateAmount);
                } while (b == i || b == a);
                do {
                    c = AIPMathUtil.generateRandom(0, candidateAmount);
                } while (c == i || c == b || c == a);
                //Napraviti xc`
                // Xc` = Xc + F * (Xb - Xa)
                NeuralNetwork xcMutated = candidates.get(i).mutate(candidates.get(a), candidates.get(b), candidates.get(c), mutationRate, crossoverProbability);
                double xcWorthy = testWorthy(xcMutated, datasetIndex), iWorthy = testWorthy(candidates.get(i), datasetIndex);
                if (xcWorthy < iWorthy) {
                    //System.out.println(Instant.now().toString() + " " + j + " evolution " + (iWorthy - xcWorthy));
                    //writeNeuronTest(xcMutated);
                    candidates.set(i, xcMutated);
                    //System.out.println(iWorthy + ",");
                    iWorthy = xcWorthy;
                    train = true;
                }
                minDelta = Math.min(minDelta, iWorthy);
            }
            epoch++;
            if (epoch % 100 == 0) {
                System.out.println("Done " + epoch);
                findAndPrintCandidate();
            }
            //System.out.println(minDelta);
        } while (train);

        AtomicReference<Double> minimumWorthynes = new AtomicReference<>(Double.MAX_VALUE);
        AtomicReference<NeuralNetwork> bc = new AtomicReference<>(candidates.get(0));
        candidates.forEach(candidate -> {
            double w = testWorthy(candidate, 1);
            System.out.println(w);
            if (w < minimumWorthynes.get()) {
                minimumWorthynes.set(w);
                bc.set(candidate);
            }
        });

        System.out.println("DONE " + minimumWorthynes.get());
        writeNeuronTest(bc.get());

        try (FileWriter fw = new FileWriter("ai.json")) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(bc.get(), fw);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void writeNeuronTest(NeuralNetwork nn) {
        System.out.println(Y[0] + "  " + nn.run(new ArrayList<>(Arrays.asList(X[0]))).toString());
        System.out.println(Y[1] + "  " + nn.run(new ArrayList<>(Arrays.asList(X[1]))).toString());
        System.out.println(Y[2] + "  " + nn.run(new ArrayList<>(Arrays.asList(X[2]))).toString());
        System.out.println(Y[3] + "  " + nn.run(new ArrayList<>(Arrays.asList(X[3]))).toString());

        System.out.println(255 + "  " + nn.run(new ArrayList<>(Arrays.asList(255D, 255D))));
        System.out.println(-255 + " " + nn.run(new ArrayList<>(Arrays.asList(-250000000D, -25000000000D))));
    }

    private void findAndPrintCandidate(){
        AtomicReference<Double> minimumWorthynes = new AtomicReference<>(Double.MAX_VALUE);
        AtomicReference<NeuralNetwork> bc = new AtomicReference<>(candidates.get(0));
        candidates.forEach(candidate -> {
            double w = testWorthy(candidate, 1);
            if (w < minimumWorthynes.get()) {
                minimumWorthynes.set(w);
                bc.set(candidate);
            }
        });
        writeNeuronTest(bc.get());
    }
}
