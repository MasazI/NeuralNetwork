package jp.masazdream.ml.nn;

import jp.masazdream.ml.graph.Graph;
import jp.masazdream.ml.inf.MachineLearning;
import jp.masazdream.ml.param.MachineLearningParameter;

/**
 * 多クラス分類 3層ニューラルネットワークの学習と識別クラス
 * 
 * @author masai
 *
 */
public class NeuralNetwork implements MachineLearning{	
	// 入力から中間層の重み係数(入力層のユニットi個、中間層のユニットj個のとき、j x i)
	double[][] mW;
	
	// 中間層から出力層への重み係数(中間層にユニット数j個、出力層のユニット数k個のとき、k x j)
    double[][] mHidden;
    
    // 入力パラメータ(ユニット)数
    int mDim;
    
    // 中間層のノード数 + 1(bias項)
    int mHiddendim;
    
    // 出力層のノード数
    int mOutputdim;
    
    // 学習回数
    final int mRepeat;
    
    // 学習係数(learning rate)
    final double mEta;

    /**
     * 起動メソッド
     * 
     * @param args
     */
    public static void main(String[] args){
    	new Graph("NeuralNetwork Backpropagation"){
    		@Override 
    		public MachineLearning createLearningMachine(){
    			return new NeuralNetwork(2, 3, 2, 15000, .3d);
    		}
    		
    		@Override
            public MachineLearning createLearningMachine(MachineLearningParameter param) {
    			NeuralNetworkParameter nnParam = (NeuralNetworkParameter) param;
                return new NeuralNetwork(nnParam.getmInputLayerNum(), 
                		nnParam.getmHiddenLayerNum(), 
                		nnParam.getmOutputLayerNum(), 
                		nnParam.getmRepeatCnt(), 
                		nnParam.getmEta());
            }
    	};    	
    }
    
    /**
     * コンストラクタ
     * 
     * @param dim 入力パラメータ数
     * @param hiddendim 中間層のノード数
     */
    public NeuralNetwork(int dim, int hiddendim, int outputdim, int repeat, double eta){
    	mDim = dim;
    	mHiddendim = hiddendim + 1; // 1はbias項
    	mOutputdim = outputdim;
    	mRepeat = repeat;
    	mEta = eta;
    	
    	// 入力層から中間層への重みの初期値を決定
    	// 中間層のbiasへは入力層からの結合なし(mHiddendim - 1)、入力層のbiasから中間層へは結合あり(mDim + 1)
        mW = new double[mHiddendim - 1][mDim + 1];
        // 中間ユニット数(iは中間ユニットのインデックス)
        for(int i = 0; i < mW.length; ++i){
        	// 入力ユニット数(jは入力ユニットのインデックス)
            for(int j = 0; j < mW[i].length; ++j){
                mW[i][j] = Math.random() * 2 - 1;
            }
        }
        
        // 中間層から出力層への重みの初期値を決定
        // 出力層はbias項なし(mOutputdim)、中間層のbiasから出力層へは結合あり(mHiddendim)
        mHidden = new double[mOutputdim][mHiddendim];
        for(int k = 0; k < mOutputdim; ++k){
	        for(int j = 0; j < mHiddendim; ++j){
	            mHidden[k][j] = Math.random() * 2 - 1;
	        }
        }
    }
    

    /**
     * 学習
     */
    @Override
    public void train(int examplesCnt, int[][] clz, double[][] examples){
    	System.out.println("[train] start");
    	// 事例データの次元
    	int exampleDim = examples[0].length;
    	
    	// bias項を追加した入力データ
    	double[][] xx = new double[examplesCnt][exampleDim + 1];
    	
    	// bias項の追加
    	for(int l = 0; l < examplesCnt; ++l){
    		// bias項の追加
    		double[] example = new double[exampleDim + 1];
            for (int j = 0; j < exampleDim; ++j) {
            	example[j + 1] = examples[l][j];
            }
            example[0] = 1;
    		
            // bias項を追加した入力データに追加
            xx[l] = example;
    	}
    	    	
        // 事例データごとの中間層のデータ
    	double[][] zz = new double[examplesCnt][mHiddendim]; 
    	
        // 事例データごとの出力層のデータ(2値分類の場合、出力層のユニットが1つであれば出力層のデータはスカラでも良い)
    	double[][] outputs = new double[examplesCnt][mOutputdim];
        
        // 学習の繰り返し
        System.out.println("Repeat Count: " + mRepeat);
        for(int n = 0; n < mRepeat; ++n){
        	// 事例リストで学習
        	for (int l = 0; l < examplesCnt; ++l) {
        		// bias項追加済みの事例
        		double[] example = xx[l];
        		
                // 中間層の出力(中間層のユニット数と同じ数)
                double[] z = new double[mHiddendim];
                
                // forward
                /*--- 入力から中間層(中間層のユニット数でループ) ---*/
                for(int j = 0; j < mW.length; ++j){
                	// 中間層への入力
                	double ui = 0;

                	// 入力データの次元(入力層にユニット数)でループ
                	for(int i = 0; i < example.length; ++i){
                		ui += mW[j][i] * example[i];
                	}
                	
                	// 中間層の出力(bias項を除いて計算)
                	z[j + 1] = sigmoid(ui);
                }
                // bias項
                z[0] = 1;
                
                // 事例ごとの中間層出力の保存
                zz[l] = z;
        	}
        	// 出力層の出力(2値分類の場合はユニットが1つで良い)
        	for (int l = 0; l < examplesCnt; ++l) {
                double out = 0;
                // 事例ごとの中間層の出力
        		double[] z = zz[l];

                /*--- 中間層から出力層(出力層x中間層のユニット数でループ) --*/
        		for(int k = 0; k < mOutputdim; ++k){
	                for(int j = 0; j < z.length; ++j){
	                	out += mHidden[k][j] * z[j];              	
	                }
	                
	                // 出力値
	                out = sigmoid(out);
	                
	                // 事例ごとの出力層出力の保存
	                outputs[l][k] = out;
        		}
        	}
        	for (int l = 0; l < examplesCnt; ++l) {
        		// 事例の出力層の出力
        		double[] out = outputs[l];
        		
        		// 事例のカテゴリー
        		int[] category = clz[l];
        		
                // backward
                /*--- 出力層から中間層 ---*/
        		double[] delta = new double[mOutputdim];
        		for(int k = 0; k < mOutputdim; ++k){
        			delta[k] = (category[k] - out[k]) * out[k] * (1 - out[k]);
                
	                // 中間層の補正値
	                double[] e = new double[mHiddendim];
	                for(int i = 0; i < mHiddendim; ++i){
	                	// 補正値(補正 x 中間層の出力)
	                	e[i] = delta[k] * zz[l][i];
	                	
	                	// 補正値に学習係数をかけて中間層から出力層への重み係数を更新する
	                	mHidden[k][i] += e[i] * mEta;
	                }
        		}
             
                /*--- 中間層から入力層 ---*/
                // 中間層の第1項はbiasであるため計算しない
                for(int i = 1; i < mHiddendim; ++i){
                	// 中間層と出力層の誤差を入力層へ伝搬する
                	double ss = 0;
                	for(int k = 0; k < mOutputdim; ++k){
	                	ss += delta[k] * mHidden[k][i];
                	}	                	
                	double sigma = ss * zz[l][i] * (1 - zz[l][i]);
                	
                	// 入力パラメータ(ユニット)数
                	for(int j = 0; j < mDim + 1; ++j){
                		mW[i - 1][j] += mEta * sigma * xx[l][j];
                	}
                }
        	}
        }
        
        // 学習完了
    	System.out.println("[train] finish");
    }
    
    /**
     * 精度測定
     */
	@Override
	public double[] evaluate(double[] data) {
		// 重み係数を利用してforwardする
		double[] example = new double[data.length + 1];
		for(int i = 0; i < data.length; ++i){
			example[i + 1] = data[i];
		}
		// bias項は常に1
		example[0] = 1;
		
		// 入力層から中間層の出力
		double[] z = new double[mHiddendim];
		// 中間層のユニット数でループ
		for(int j = 0; j < mW.length; ++j){
			double ui = 0;
			// 入力層のユニット数でループ
			for(int i = 0; i < example.length; ++i){
				ui += example[i] * mW[j][i];
			}
			z[j + 1] = sigmoid(ui);
		}
		// bias項は1
		z[0] = 1;
		
		// 中間層から出力層の出力
		double[] outs = new double[mOutputdim];
		// 出力層のユニット数でループ(mHidden.lengthはmOutputdimと等価)
		for(int k = 0; k < mHidden.length; ++k){
			double out = 0;
			for(int j = 0; j < z.length; ++j){
				out += mHidden[k][j] * z[j];
			}
			outs[k] = sigmoid(out);
		}
		
		return outs;
	}
	
	/**
	 * シグモイド関数
	 * 
	 * @param input
	 * @return
	 */
	private double sigmoid(double input){
		return 1 / (1 + Math.exp(-input));
	}
}
