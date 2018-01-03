package TrainingSTreams;

import activationFunctions.*;
import gameMechanics.Board;
import gameMechanics.Decision;
import neuralNet.MLNN;
import rules.Rules;
import strategy.BasicStrategyInputStream;
import strategy.ConsulInputStream;
import strategy.InputStream;
import strategy.MLNNInputStream;

public class BJTrainingStream {
	Board mainBoard;
	InputStream idealStrategy;
	double[] bets = {1.0, 1.0, 1.0, 1.0};
	boolean[] insuranceBets = {false, false, false, false};
	boolean textOn=false;
	Rules gameRules;
	
	public BJTrainingStream(Board mainBoard, InputStream idealStrategy, Rules gameRules) {
		this.mainBoard=mainBoard;
		this.idealStrategy=idealStrategy;
		this.gameRules=gameRules;
	}
	
	public double[] getInputs() {

		mainBoard.roundOne(bets, textOn);
		if (gameRules.INSURANCE && mainBoard.insuranceCheck()) {
			mainBoard.roundTwo(insuranceBets, textOn);
		}

		if (mainBoard.dealerBJRound()) {
			mainBoard.startRoundThree();
			while (mainBoard.round == 3) {
				}
				if (trainingOn) {
					trainNN(NNinput, mainBoard.boardMatrix(simpleTraining),
							inputStream.round3decisionMatrix(mainBoard));
				}

				// if false 10 times, stand on hand.

			}
		}
	}

	

	public double[] getIdealOutputs() {
		System.out.println("Error in get getIdealOutputs");
		return null;
	}

	public void nextCase() {
	
	}

	public void truthStart() {
	}

	public boolean truthEnd() {
		System.out.println("Error in get truthend");
		return false;
	}

	public double[] nextTruthInput() {
		System.out.println("Error in get nextTruthInput");
		return null;
	}

	public double[] nextTurtIdealOutput() {
		System.out.println("Error in get nextTurtIdealOutput");
		return null;
	}

	public void nextTruthCase() {
	}
}
