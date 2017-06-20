package cn.neuron;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

public class NeuronNetworkConstruct implements LearningEventListener {
	
	public static void main(String[] args) {
		new NeuronNetworkConstruct().Calculate();
	}

	/***
	 * ����������AND��OR�߼�����
	 */
	public void Calculate() {
		// ����ѵ����������������һ�����
		DataSet trainingSet = new DataSet(2, 1);
		trainingSet.addRow(new DataSetRow(new double[] { 0, 0 }, new double[] { 0 }));
		trainingSet.addRow(new DataSetRow(new double[] { 0, 1 }, new double[] { 1 }));
		trainingSet.addRow(new DataSetRow(new double[] { 1, 0 }, new double[] { 1 }));
		trainingSet.addRow(new DataSetRow(new double[] { 1, 1 }, new double[] { 1 }));
		// ����һ����֪������������̼���2������֪�������1�����������Neuroph�ṩ��Perceptron��
		NeuralNetwork myPerceptron = new Perceptron(2, 1);
		// ��������
		myPerceptron.setLearningRule(new BackPropagation());		
		LearningRule lr = myPerceptron.getLearningRule();
		lr.addListener(this);
		// ��ʼѧϰѵ����
		myPerceptron.learn(trainingSet);
		// ���Ը�֪���Ƿ���ȷ�������ӡ
		System.out.println("Testing trained perceptron");
		testNeralNetwork(myPerceptron, trainingSet);
	}

	/***
	 * ˫��������XOR�߼�����
	 */
	public void XORCalculate() {
		// ����ѵ����������������һ�����
		DataSet trainingSet = new DataSet(2, 1);
		trainingSet.addRow(new DataSetRow(new double[] { 0, 0 }, new double[] { 0 }));
		trainingSet.addRow(new DataSetRow(new double[] { 0, 1 }, new double[] { 1 }));
		trainingSet.addRow(new DataSetRow(new double[] { 1, 0 }, new double[] { 1 }));
		trainingSet.addRow(new DataSetRow(new double[] { 1, 1 }, new double[] { 0 }));
		//���������֪���������2����Ԫ��������3����Ԫ����������Ϊ1��������Ԫ������ʹ��TANH���亯���������ĸ�ʽ�����
		MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron(TransferFunctionType.TANH,2,2,1);
		//��������
		myMlPerceptron.setLearningRule(new BackPropagation());
		LearningRule learningRule = myMlPerceptron.getLearningRule();
        learningRule.addListener(this);
		//��ʼѵ��
		myMlPerceptron.learn(trainingSet);
		// ���Ը�֪���Ƿ���ȷ�������ӡ
		System.out.println("Testing trained perceptron");
		testNeralNetwork(myMlPerceptron, trainingSet);
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
