package NNEnvironments;


import TrainingSTreams.DoubleArrayTrainingStream;
import activationFunctions.ActivationFunction;
import activationFunctions.Maxout;
import activationFunctions.Relu;
import activationFunctions.Sigmoid;
import neuralNet.MLNN;
import neuralNet.MLNNThread;

public class Mind extends NNEnviormentAbstract{
	
	/**
	 * Evolutionary mind is a collection of Neural nets that all are trying to solve a function, with a final output that is
	 * based on a function of the invidual NN outputs.  So if 3/5 outputs say 1, and 2/5 say 0, the final output would see that 1 
	 * is more likely, and so output 1.  Feedback training is done both on
	 */

	public Mind() {

			
		}
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
		
		MLNNThread testThread = new MLNNThread(test, 10000, true, true, 100, nnTrainingStream);
				
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
		
		testThread = new MLNNThread(test2, 10000, true, true, 100, nnTrainingStream);
	
		System.out.println(" Recall:");
		testThread.printTruthTable();
		testThread.printTruthTableIdeal();
	}
}
