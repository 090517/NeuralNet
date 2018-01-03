package TrainingSTreams;

public abstract class TrainingStreamAbstract {
	// creates a stream of cases for bots to train against.

	public TrainingStreamAbstract() {
	}

	public double[] getInputs() {
		System.out.println("Error in get inputs");
		return null;
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
