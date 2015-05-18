package jp.masazdream.ml.nn;

import jp.masazdream.ml.param.MachineLearningParameter;

public class NeuralNetworkParameter implements MachineLearningParameter{
	private int mInputLayerNum;
	
	private int mHiddenLayerNum;
	
	private int mOutputLayerNum;
	
	private int mRepeatCnt;
	
	private double mEta;

	/**
	 * ニューラルネットワークパラメータ
	 * 
	 * @param inputLayerNum 入力層のユニット数
	 * @param hiddenLayerNum 中間層のユニット数
	 * @param outputLayerNum 出力層のユニット数
	 * @param repeatCnt リピート回数
	 * @param eta 学習係数
	 */
	public NeuralNetworkParameter(int inputLayerNum, int hiddenLayerNum, int outputLayerNum, int repeatCnt, double eta){
		mInputLayerNum = inputLayerNum;
		mHiddenLayerNum = hiddenLayerNum;
		mOutputLayerNum = outputLayerNum;
		mRepeatCnt = repeatCnt;
		mEta = eta;
	}
	
	public int getmInputLayerNum() {
		return mInputLayerNum;
	}

	public int getmHiddenLayerNum() {
		return mHiddenLayerNum;
	}

	public int getmOutputLayerNum() {
		return mOutputLayerNum;
	}

	public int getmRepeatCnt() {
		return mRepeatCnt;
	}

	public double getmEta() {
		return mEta;
	}
	
	
}
