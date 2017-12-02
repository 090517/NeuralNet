package neuralNet;

import activationFunctions.ActivationFunction;
import activationFunctions.Relu;
import activationFunctions.Sigmoid;

public class XORTraining {
	public static void main(String[] args) {

		double xorIN[][] = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };

		double xorExpected[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };
		
		BJNeuralNet xorNN= new BJNeuralNet(2,3,1,0.7,0.9);
		
		for (int runCnt=0; runCnt<10000; runCnt++) {
			for(int loc=0; loc<xorIN.length; loc++) {
				//System.out.println(Double.toString(xorIN[loc][0])+Double.toString(xorIN[loc][1]));
				xorNN.calOuput(xorIN[loc]);
				xorNN.calcError(xorExpected[loc]);
				xorNN.train();
			}
			
			//System.out.println("Trial #" + runCnt + " Error:" + xorNN.getError());
			
		}
		
		System.out.println( "Error:" + xorNN.getError()+ " Recall:");

		for (int i=0;i<xorIN.length;i++) {

			for (int j=0;j<xorIN[0].length;j++) {
				System.out.print( xorIN[i][j] +":" );
			}

			double out[] = xorNN.calOuput(xorIN[i]);
			System.out.println("="+out[0]);
		}
		
		
		int hiddenLayers=3;
		
		int[] hiddenArray=new int[hiddenLayers];
		hiddenArray[0]=100;
		hiddenArray[1]=100;
		hiddenArray[2]=100;
		
		ActivationFunction[] AF=new ActivationFunction[hiddenLayers+1];

		for (int i = 0; i < AF.length; i++) {
			AF[i] = new Relu();
		}
		AF[hiddenLayers] = new Relu();

		boolean[] bias = new boolean[hiddenLayers+1];
		for (int i = 0; i < AF.length; i++) {
			bias[i] = true;
		}
				
		//int[] nnHiddenArrangment, int inputs, int outputs, ActivationFunction[] AFArray, boolean[] bias
		MLNN test=new MLNN(hiddenArray, 2, 1, AF, bias, .1);
				
		System.out.println(" Recall:");

		for (int i=0;i<xorIN.length;i++) {
			for (int j=0;j<xorIN[0].length;j++) {
				System.out.print( xorIN[i][j] +":" );
			}
			double out[] = test.forwardProp(xorIN[i]);
			System.out.println("="+out[0]);
		}
		
		for (int i=0; i<10000; i++) {
			for (int j=0; j<xorIN.length; j++) {
				test.forwardProp(xorIN[j]);
				test.backProp(xorExpected[j]);
			}
		}
		
		System.out.println(" Recall:");

		for (int i=0;i<xorIN.length;i++) {
			for (int j=0;j<xorIN[0].length;j++) {
				System.out.print( xorIN[i][j] +":" );
			}
			double out[] = test.forwardProp(xorIN[i]);
			System.out.println("="+out[0]);
		}
		
		test.saveNN("nn1");
		MLNN test2=MLNN.readNN("nn1");
		
		System.out.println(" Recall:");

		for (int i=0;i<xorIN.length;i++) {
			for (int j=0;j<xorIN[0].length;j++) {
				System.out.print( xorIN[i][j] +":" );
			}
			double out[] = test2.forwardProp(xorIN[i]);
			System.out.println("="+out[0]);
		}
		
	}
}
