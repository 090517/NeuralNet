package TESTSCRIPTS;

import activationFunctions.ActivationFunction;
import activationFunctions.Sigmoid;
import neuralNet.MLNN;

//Follows the example at https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

public class testTraining {
	public static void main(String[] args) {
	int[] hidden= {2};
	int inputs=2;
	int outputs=2;
	boolean[] bias= {true,true};
	ActivationFunction[] AF= {new Sigmoid(), new Sigmoid()};
	

	//public MLNN(int[] nnHiddenArrangment, int inputs, int outputs, ActivationFunction[] AFArray, boolean[] bias, double newLearningRate)
	MLNN testNN = new MLNN(hidden, inputs, outputs, AF, bias, .5);
	
	System.out.println("---Random weights---");
	testNN.printNN();
	
	double[][] layer1Weights= {{.15,.20,.35},{.25,.30,.35}};
	double[][] layer2Weights= {{.40,.45,.6},{.50,.55,.60}};
	
	testNN.setWeights(0, layer1Weights);
	testNN.setWeights(1, layer2Weights);
	
	System.out.println("---Set weights---");
	testNN.printNN();
	
	double[] input1= {.05,.10};
	double[] ideal= {.01, .99};
	
	System.out.println("---First Forward Prop---");	
	double[] output=testNN.forwardPropWError(input1, ideal);
	System.out.println(output[0]);
	System.out.println(output[1]);
	
	System.out.println("---First Error Rate---");
	System.out.println(testNN.getError());
	
	testNN.backProp(ideal);
	testNN.printNN();
	
	for (int i=0; i<10000; i++) {
		testNN.forwardPropWError(input1, ideal);
		testNN.backProp(ideal);
	}
	
	System.out.println();
	testNN.printNN();
	System.out.println("Final " + testNN.forwardProp(input1)[0]);
	System.out.println("Final " + testNN.forwardProp(input1)[1]);
	testNN.plotErrors();
	}
}
