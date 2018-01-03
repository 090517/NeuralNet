package TESTSCRIPTS;

import activationFunctions.ActivationFunction;
import activationFunctions.Maxout;
import activationFunctions.Relu;
import activationFunctions.Sigmoid;
import neuralNet.MLNN;

public class XORmomenutmVSNon {

	public static void main(String[] args) {

		double xorIN[][] = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };
		double xorExpected[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

		int hiddenLayers = 2;
		int[] hiddenArray = new int[hiddenLayers];
		hiddenArray[0] = 2;
		hiddenArray[1] = 2;
		// hiddenArray[2]=1;

		ActivationFunction[] AF = new ActivationFunction[hiddenLayers + 1];
		for (int i = 0; i < AF.length; i++) {
			AF[i] = new Sigmoid();
		}
		// AF[2] = new Maxout();

		boolean[] bias = new boolean[hiddenLayers + 1];
		for (int i = 0; i < AF.length; i++) {
			bias[i] = true;
		}

		MLNN test = new MLNN(hiddenArray, 2, 1, AF, bias, 1);
		MLNN testWMomentuM = new MLNN(hiddenArray, 2, 1, AF, bias, .5, .5);


		// test.printNN();
		
		int countFailed = 0;
		int sampleSize = 10000;

		int startRounds=1000;
		int endRounds=2000000;
		int margin=1000;
		long startime = System.currentTimeMillis();

		System.out.println("Without momentum");

		for (int trainingRounds = startRounds; trainingRounds < endRounds; trainingRounds = trainingRounds + margin) {

			startime = System.currentTimeMillis();
			countFailed = 0;
			for (int rounds = 0; rounds < sampleSize; rounds++) {
				for (int i = 0; i < trainingRounds; i++) {
					for (int j = 0; j < xorIN.length; j++) {
						test.forwardPropWError(xorIN[j], xorExpected[j]);
						test.backProp(xorExpected[j]);
					}
					if (i % 100 == 0) {
						if (test.getError() < .001) {
							break;
						}
					}
					if (i == trainingRounds - 1) {
						countFailed++;
					}
				}
				test.resetNN();
			}

			System.out
					.println(trainingRounds+ "\t" + countFailed + "\t" + (System.currentTimeMillis() - startime));
			

		}
		System.out.println("With momentum");
		for (int trainingRounds = startRounds; trainingRounds < endRounds; trainingRounds = trainingRounds + margin) {
			startime = System.currentTimeMillis();
			countFailed = 0;
			for (int rounds = 0; rounds < sampleSize; rounds++) {
				for (int i = 0; i < trainingRounds; i++) {
					for (int j = 0; j < xorIN.length; j++) {
						testWMomentuM.forwardPropWError(xorIN[j], xorExpected[j]);
						testWMomentuM.backProp(xorExpected[j]);
					}
					if (i % 100 == 0) {
						if (testWMomentuM.getError() < .001) {
							break;
						}
					}
					if (i == trainingRounds - 1) {
						countFailed++;
					}
				}
				testWMomentuM.resetNN();
			}	
			System.out
			.println(trainingRounds+ "\t" + countFailed + "\t" + (System.currentTimeMillis() - startime));
		}
	}
}
