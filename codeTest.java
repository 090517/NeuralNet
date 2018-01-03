import activationFunctions.ActivationFunction;
import activationFunctions.LeakyRelu;
import activationFunctions.Relu;
import activationFunctions.Sigmoid;
import activationFunctions.TanH;
import neuralNet.MLNN;

public class codeTest {
	public static void main(String[] arg) {
		double[][] weights={ { 0.15, 0.20, .35 }, { 0.25, 0.3, .35 } };
		double[][] weights2={ { 0.4, 0.45, .6 }, { 0.5, 0.55, .6 } };
		double[] ideal= {.01, .99};
		double input[]=new double[2];
		input[0]=.05;
		input[1]=.10;
		
		
		int hiddenLayers=3;
		int[] hiddenArray=new int[hiddenLayers];
		hiddenArray[0]=200;
		hiddenArray[0]=200;
		hiddenArray[0]=200;
		
		double reluRate=.01;
		ActivationFunction[] AF=new ActivationFunction[hiddenLayers+1];
		//AF[0]=new LeakyRelu(reluRate);
		//AF[1]=new LeakyRelu(reluRate);
		//AF[2]=new LeakyRelu(reluRate);
		//AF[3]=new LeakyRelu(reluRate);
		AF[0]=new TanH();
		AF[1]=new TanH();
		AF[2]=new TanH();
		AF[3]=new TanH();
		
		boolean[] bias=new boolean[hiddenLayers+1];
		for (int i = 0; i < AF.length; i++) {
			bias[i] = true;
		}
		
		//int[] nnHiddenArrangment, int inputs, int outputs, ActivationFunction[] AFArray, boolean[] bias
		MLNN test=new MLNN(hiddenArray, 2, 2, AF, bias, .5);
		
		//test.setWeights(1, weights);
		//test.setWeights(2, weights2);
		
		double[] output=test.forwardPropWError(input, ideal);
		System.out.println(output[0]+" "+output[1]);		
		System.out.println("Error:"+test.getError());
		System.out.println();

		test.backProp(ideal);
				
		output=test.forwardPropWError(input, output);
		System.out.println(output[0]+" "+output[1]);		
		System.out.println("Error:"+test.getError());
		
		for (int i=0; i<1000; i++) {
			output=test.forwardPropWError(input, output);
			System.out.println(output[0]+" "+output[1]);		
			System.out.println("Error:"+test.getError());
			System.out.println();

			test.backProp(ideal);
		}
		
		System.out.println("---------FINAL---------");
		System.out.println(output[0]+" "+output[1]);		
		System.out.println("Error:"+test.getError());
		
	}
}
