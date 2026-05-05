import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import firebaseConfig from '../../firebase-applet-config.json';

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app, firebaseConfig.firestoreDatabaseId);

// 强制的错误处理封装
export function handleFirestoreError(error: unknown, operation: string, path: string) {
  console.error(`Firebase Error [${operation}] at ${path}:`, error);
  throw new Error(JSON.stringify({
    error: error instanceof Error ? error.message : String(error),
    operation,
    path
  }));
}
