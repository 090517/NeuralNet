package TESTSCRIPTS;

import java.util.concurrent.TimeUnit;

import TrainingSTreams.DoubleArrayTrainingStream;
import activationFunctions.ActivationFunction;
import activationFunctions.LeakyRelu;
import activationFunctions.Relu;
import activationFunctions.Sigmoid;
import activationFunctions.TanH;
import neuralNet.MLNN;

public class localOptimumTest {
	static double xorIN[][] = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };
	static double xorExpected[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

	public static void main(String[] args) throws Exception {
		
		DoubleArrayTrainingStream nnTrainingStream=new DoubleArrayTrainingStream(xorIN, xorExpected);
		
		//Parrameters;
		int[] layersToTest = { 1, 2, 3, 5};
		int[] neuronsToTest = { 2, 3, 4, 6, 10, 20, 100};
		ActivationFunction[] activationsToTest = { new Sigmoid(), new TanH(), new LeakyRelu(.1), new Relu()};
		boolean[] biasToTest = { true, false };
		double[] learningRateToTest = { .001, .01, .05, .1, .2, .5, 1 };

		long startTime = System.currentTimeMillis();
		int hiddenLayers = 1;
		int[] hiddenArray = new int[hiddenLayers];
		ActivationFunction[] AF = new ActivationFunction[hiddenLayers + 1];
		boolean[] bias = new boolean[hiddenLayers + 1];
		for (int i = 0; i <= hiddenLayers; i++) {
			if (i != hiddenLayers) {
				hiddenArray[i] = neuronsToTest[0];
			}
			AF[i] = activationsToTest[0];
			bias[i] = biasToTest[0];
		}
		double learningRate = learningRateToTest[5];
		MLNN testNN = new MLNN(hiddenArray, 2, 1, AF, bias, learningRate);
		long endTime = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		
			
		double trials = 100;
		int trainingEpochs = 15;
		
		
		// one hidden layer
		for (int layerI = 0; layerI < layersToTest.length; layerI++) {
			for (int neuronI = 0; neuronI < neuronsToTest.length; neuronI++) {
				for (int afI = 0; afI < activationsToTest.length; afI++) {
					for (int biasI = 0; biasI < biasToTest.length; biasI++) {
						for (int learningI = 0; learningI < learningRateToTest.length; learningI++) {
							startTime = System.currentTimeMillis();
							hiddenLayers = layersToTest[layerI];

							hiddenArray = new int[hiddenLayers];
							AF = new ActivationFunction[hiddenLayers + 1];
							bias = new boolean[hiddenLayers + 1];

							for (int i = 0; i <= hiddenLayers; i++) {
								if (i != hiddenLayers) {
									hiddenArray[i] = neuronsToTest[neuronI];
								}
								AF[i] = activationsToTest[afI];
								bias[i] = biasToTest[biasI];
							}

							learningRate = learningRateToTest[learningI];

							testNN = new MLNN(hiddenArray, 2, 1, AF, bias, learningRate);
						
							System.out.print(
									"Layers:" + layersToTest[layerI] + "\tNeurons:" + neuronsToTest[neuronI]
									+ "\t" + activationsToTest[afI].getClass().getName() + "\tBias:" + biasToTest[biasI]
									+ "\tlearningRate:" + learningRateToTest[learningI] + "\tPercentage Failed:"
									+ trialNN(testNN, trials, trainingEpochs, nnTrainingStream));
							
							endTime = System.currentTimeMillis();
							totalTime = endTime - startTime;
							System.out.println("\tTime:" + totalTime/1000.0 + " Seconds");
							
						}
					}
				}
			}
		}

	}

	public static void printOutput(MLNN inputNN) throws Exception {
		System.out.println(" Recall:");
		for (int i = 0; i < xorIN.length; i++) {
			for (int j = 0; j < xorIN[0].length; j++) {
				System.out.print(xorIN[i][j] + ":");
			}
			double out[] = inputNN.forwardProp(xorIN[i]);
			System.out.println("=" + out[0]);
		}
	}

	public static boolean testNN(MLNN inputNN) throws Exception {
		for (int i = 0; i < xorIN.length; i++) {
			double out[] = inputNN.forwardProp(xorIN[i]);
			if ((out[0] - xorExpected[i][0]) > .15) {
				return false;
			}
		}
		return true;
	}

	public static double trialNN(MLNN inputNN, double Trials, int TrainingEpochs, DoubleArrayTrainingStream nnTrainingStream) throws Exception {

		double falseCount = 0;
		for (int NNTrials = 0; NNTrials < Trials; NNTrials++) {
			for (int i = 0; i < TrainingEpochs; i++) {
				inputNN.forwardPropMulti(nnTrainingStream.getInputs());
				inputNN.backProp(nnTrainingStream.getIdealOutputs());
				nnTrainingStream.nextCase();
			}
			if (!testNN(inputNN)) {
				falseCount++;
			}
			inputNN.resetNN();
		}
		return falseCount/Trials;
	}

}
