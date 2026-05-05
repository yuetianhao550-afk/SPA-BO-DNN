import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { OptimizerService } from './src/services/optimizer.js';
import { HardwareGuard } from './src/services/hardware.js';

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  const optimizer = new OptimizerService();

  // API routes
  app.get("/api/sensors", (req, res) => {
    res.json({
      algaeDensity: { value: Math.random() * 1000 + 4000, history: Array.from({length: 20}, () => Math.random() * 1000 + 4000) },
      ph: { value: 7.2 + Math.random() * 1.5, history: Array.from({length: 20}, () => 7 + Math.random() * 2) },
      temperature: { value: 20 + Math.random() * 10, history: Array.from({length: 20}, () => 20 + Math.random() * 10) },
      conductivity: { value: 30 + Math.random() * 20, history: Array.from({length: 20}, () => 30 + Math.random() * 20) }
    });
  });

  app.post("/api/hardware/control", async (req, res) => {
    try {
      const command = req.body;
      console.log("Automation System executing hardware command:", command);
      res.json({ success: true, status: 'EXECUTING', ...command });
    } catch (error) {
      res.status(400).json({ error: (error as Error).message });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
