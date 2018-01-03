package TESTSCRIPTS;

import activationFunctions.ActivationFunction;
import activationFunctions.Maxout;
import activationFunctions.Sigmoid;
import neuralNet.MLNN;

public class testTrainingAddedLayer {


	public static void main(String[] args) {
		
	//Follows the example at https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
	//public MLNN(int[] nnHiddenArrangment, int inputs, int outputs, ActivationFunction[] AFArray, boolean[] bias, double newLearningRate)
	int[] hidden= {2,2};
	int inputs=2;
	int outputs=2;
	boolean[] bias= {true,true, false};
	ActivationFunction[] AF= {new Sigmoid(), new Sigmoid(), new Maxout()};
	
	MLNN testNN = new MLNN(hidden, inputs, outputs, AF, bias, .5);
	
	System.out.println("---Random weights---");
	testNN.printNN();
	
	double[][] layer1Weights= {{.15,.20,.35},{.25,.30,.35}};
	double[][] layer2Weights= {{.40,.45,.6},{.50,.55,.60}};
	double[][] layer3Weights= {{1.0,1.0}};
	
	testNN.setWeights(0, layer1Weights);
	testNN.setWeights(1, layer2Weights);
	testNN.setWeights(2, layer3Weights);
	
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
	
	System.out.println("---First backprop---");

	testNN.backProp(ideal);
	testNN.printNN();
	
	
	for (int i=0; i<10; i++) {
		testNN.forwardPropWError(input1, ideal);
		testNN.backProp(ideal);
	}
	
	System.out.println("---After Training weights---");
	
	System.out.println();
	testNN.printNN();
	System.out.println("Final " + testNN.forwardProp(input1)[0]);
	System.out.println("Final " + testNN.forwardProp(input1)[1]);
	
	//public void setWeights(int layer, double[][] weights)
	}

}
