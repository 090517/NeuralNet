package TESTSCRIPTS;

import TrainingSTreams.DoubleArrayTrainingStream;
import activationFunctions.ActivationFunction;
import activationFunctions.Maxout;
import activationFunctions.Relu;
import activationFunctions.Sigmoid;
import neuralNet.MLNN;
import neuralNet.MLNNThread;

public class XORTraining {
	public static void main(String[] args) {

		double xorIN[][] = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };
		double xorExpected[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };
		DoubleArrayTrainingStream nnTrainingStream=new DoubleArrayTrainingStream(xorIN, xorExpected);
		
		int hiddenLayers=2;	
		int[] hiddenArray=new int[hiddenLayers];
		hiddenArray[0]=2;
		hiddenArray[1]=2;
		
		ActivationFunction[] AF=new ActivationFunction[hiddenLayers+1];
		for (int i = 0; i < AF.length; i++) {
			AF[i] = new Sigmoid();
		}

		boolean[] bias = new boolean[hiddenLayers+1];
		for (int i = 0; i < AF.length; i++) {
			bias[i] = true;
		}
				
		MLNN test=new MLNN(hiddenArray, 2, 1, AF, bias, 1);
		
		MLNNThread testThread = new MLNNThread(test, 10000, true, true, 1, nnTrainingStream);
				
		System.out.println(" Recall:");
		testThread.printTruthTable();
		testThread.printTruthTableIdeal();
		
		//test.printNN();
		testThread.run();	
		testThread.plotErrors();
		
		//test.printNN();
		
		System.out.println(" Recall:");
		testThread.printTruthTable();
		testThread.printTruthTableIdeal();
		
		test.saveNN("nn1");
		MLNN test2=MLNN.readNN("nn1");
		
		System.out.println(" Recall:");
		testThread.printTruthTable();
		testThread.printTruthTableIdeal();
	}
}
