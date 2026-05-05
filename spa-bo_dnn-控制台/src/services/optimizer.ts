import * as tf from '@tensorflow/tfjs-node';
import { db, handleFirestoreError } from '../lib/firebase.js';
import { collection, addDoc } from 'firebase/firestore';

export interface TreatmentParams {
  energyDensity: number; // 复合特征
  藻含量: number;
  电容级数: number;
}

// 代理模型：DNN (128-64-32)
const createModel = async () => {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [3] }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  return model;
};

export class OptimizerService {
  private model: tf.LayersModel | null = null;

  async init() {
    this.model = await createModel();
  }

  async predict(params: TreatmentParams): Promise<number> {
    if (!this.model) await this.init();
    const input = tf.tensor2d([[params.energyDensity, params.藻含量, params.电容级数]]);
    const prediction = this.model!.predict(input) as tf.Tensor;
    const result = (await prediction.data())[0];
    input.dispose();
    prediction.dispose();

    // 将推荐结果存储到 Firebase
    try {
      await addDoc(collection(db, 'experiments'), {
        ...params,
        resultScore: result,
        timestamp: new Date().toISOString()
      });
    } catch (e) {
      handleFirestoreError(e, 'create', 'experiments');
    }
    
    return result;
  }
}
