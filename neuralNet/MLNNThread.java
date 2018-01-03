package neuralNet;

import TrainingSTreams.DoubleArrayTrainingStream;

//objective of this class is to have a MLNN wrapper than can be run in a thread
public class MLNNThread implements Runnable {
	MLNN mainNN;
	int numberOfTrainingRounds;
	DoubleArrayTrainingStream NNTrainingStream;
	boolean errorRecording;
	boolean errorPrinting;
	int recordingInterval;

	public MLNNThread(MLNN newMainNN, int newNumberOfTrainingRounds, boolean newErrorRecording,
			boolean newErrorPrinting, int recordingInterval, DoubleArrayTrainingStream NNTrainingStream) {
		this.mainNN = newMainNN;
		this.numberOfTrainingRounds = newNumberOfTrainingRounds;
		this.NNTrainingStream = NNTrainingStream;
		this.errorRecording = newErrorRecording;
		this.errorPrinting = newErrorPrinting;
		this.recordingInterval = recordingInterval;
	}

	@Override
	public void run() {
		if (errorRecording) {
			for (int i = 0; i < numberOfTrainingRounds; i++) {
				mainNN.forwardPropWError(NNTrainingStream.getInputs(), NNTrainingStream.getIdealOutputs());
				mainNN.backProp(NNTrainingStream.getIdealOutputs());
				if ((i + 1) % recordingInterval == 0) {
					if (errorPrinting) {
						System.out.println("Round:" + (i+1) + "\tError:" + mainNN.getError());
					} else
						mainNN.getError();
				}
				NNTrainingStream.nextCase();
			}
		} else {
			for (int i = 0; i < numberOfTrainingRounds; i++) {
				mainNN.forwardProp(NNTrainingStream.getInputs());
				mainNN.backProp(NNTrainingStream.getIdealOutputs());
			}
		}
	}

	public double[] MLNNThreadForwardProp(double[] input) {
		return mainNN.forwardProp(input);
	}

	public void plotErrors() {
		mainNN.plotErrors();
	}

	public void printTruthTable() {
		NNTrainingStream.truthStart();
		System.out.println("Truth Table Actual:");
		NNTrainingStream.truthStart();
		while(!NNTrainingStream.truthEnd()) {
			System.out.println("Input: " + doubleArrayToString(NNTrainingStream.nextTruthInput()) + "Output:" + doubleArrayToString(mainNN.forwardProp(NNTrainingStream.nextTruthInput())));
			NNTrainingStream.nextTruthCase();
		}
	}
	
	public void printTruthTableIdeal() {
		NNTrainingStream.truthStart();
		System.out.println("Truth Table Ideal");
		NNTrainingStream.truthStart();
		while(!NNTrainingStream.truthEnd()) {
			System.out.println("Input: " + doubleArrayToString(NNTrainingStream.nextTruthInput()) + "Output:" + doubleArrayToString(mainNN.forwardProp(NNTrainingStream.nextTruthInput())));
			NNTrainingStream.nextTruthCase();
		}
	}

	private String doubleArrayToString(double[] input) {
		String output = new String();
		for (int i = 0; i < input.length; i++) {
			if (i == input.length - 1) {
				output = output.concat(input[i]+"\t");
			} else
				output = output.concat(input[i] + ",");
		}
		return output;
	}
}
