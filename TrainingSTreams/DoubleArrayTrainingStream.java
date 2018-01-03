package TrainingSTreams;

public class DoubleArrayTrainingStream extends TrainingStreamAbstract {
	private double[][] inputs;
	private double[][] idealOutputs;
	private int currentIndex;
	boolean truthDone;
	int truthIndex;
	
	//creates a stream of cases for bots to train against.
	
	public DoubleArrayTrainingStream(double[][] newInputs, double[][] newIdealOutputs) {
		inputs=newInputs;
		this.idealOutputs=newIdealOutputs;
		currentIndex=0;
		if (inputs.length!=idealOutputs.length) {
			System.out.println("inputs and outputs sizes don't match");
		}
		truthDone=false;
		truthIndex=0;
	}
	
	public double[] getInputs() {
		return inputs[currentIndex];
	}
	
	public double[] getIdealOutputs() {
		return idealOutputs[currentIndex];
	}
	
	public void nextCase(){
		if (currentIndex==inputs.length-1) {
			currentIndex=0;
		}
		else currentIndex++;
	}
		
	public void truthStart() {
		truthDone=false;
		truthIndex=0;
	}
	
	public boolean truthEnd() {
		if (truthIndex==(inputs.length))
			truthDone=true;
		return truthDone;
	}
	
	public double[] nextTruthInput() {
		return inputs[truthIndex];
	}
	
	public double[] nextTurtIdealOutput() {
		return idealOutputs[truthIndex];
	}
	
	public void nextTruthCase() {
		truthIndex++;
	}	
}
