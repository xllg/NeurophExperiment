package cn.neuron;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;

public class AnimalClassic implements LearningEventListener {
	public static void main(String[] args) {
		new AnimalClassic().Calculate();
	}
	
	public void Calculate() {
		//����animalѵ������
		String trainingSetFileName = "data_sets/animals_data.txt";		
		//�������������ֱ�Ϊ�������Ԫ����20�������Ԫ����7��
		int inputsCount = 20;
		int outputsCount = 7;
		DataSet dataSet = DataSet.createFromFile(trainingSetFileName, inputsCount, outputsCount, "\t");
		// ����ѵ����
		System.out.println("Creating neural network...");
		// ����һ�������磬����һ�����㣬��Ԫ��Ϊ22
		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount,16,outputsCount);
		// ����ѧϰ����BP
		MomentumBackpropagation learningRule = (MomentumBackpropagation)neuralNet.getLearningRule();
		learningRule.addListener(this);
		//���������ѧϰ�ٶ�
		learningRule.setLearningRate(0.2);
		learningRule.setMaxError(0.01);
		System.out.println("Training network...");
		// ��ʼѧϰѵ����
		neuralNet.learn(dataSet);
		System.out.println("Training completed.");
		// ���Ը�֪���Ƿ���ȷ�������ӡ
		System.out.println("Testing trained network...");
		testNeralNetwork(neuralNet, dataSet);
	}
	
	/***
	 * �������
	 * @param myPerceptron
	 * @param trainingSet
	 */
	public static void testNeralNetwork(NeuralNetwork myPerceptron,
			DataSet trainingSet) {
		for (DataSetRow dataRow : trainingSet.getRows()) {
			myPerceptron.setInput(dataRow.getInput());
			myPerceptron.calculate();
			double[] networkOutput = myPerceptron.getOutput();
			System.out.print("Input: " + Arrays.toString(dataRow.getInput()));
			System.out.println(" Output: " + Arrays.toString(networkOutput));
		}
	}
	
	/***
	 * �¼�����
	 */
	@Override
	public void handleLearningEvent(LearningEvent event) {
		BackPropagation bp = (BackPropagation)event.getSource();
		System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError());
	}
}
