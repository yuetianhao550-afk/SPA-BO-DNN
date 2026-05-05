import { useState, useEffect, useRef } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, AreaChart, Area, ComposedChart, Bar, Legend,
  ScatterChart, Scatter, ZAxis, Cell
} from 'recharts';
import { Zap, Bot, Settings2, Activity, ShieldAlert, Cpu, Droplets, Thermometer, Zap as ZapIcon, Info, Network, Terminal as TerminalIcon } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';

export default function App() {
  const [automationActive, setAutomationActive] = useState(false);
  const [hardware, setHardware] = useState({ ballGap: 12.0, tungstenGap: 0.8, capacitorLevel: 3 });
  
  const [history, setHistory] = useState<any[]>([]);
  const [lossData, setLossData] = useState<{epoch: number, loss: number}[]>([]);
  const [scatterData, setScatterData] = useState<{b: number, t: number, s: number}[]>([]);
  const [logs, setLogs] = useState<{time: string, msg: string, type: 'info'|'warn'|'success'|'system'}[]>([
    {time: new Date().toISOString().substring(11,19), msg: '系统已启动，等待连接物理设备或启用孪生沙盒...', type: 'system'}
  ]);

  const addLog = (msg: string, type: 'info'|'warn'|'success'|'system' = 'info') => {
    const time = new Date().toISOString().substring(11, 19);
    setLogs(prev => [...prev.slice(-49), { time, msg, type }]);
  };

  // Refs for the physics and DNN interval
  const autoRef = useRef(automationActive);
  autoRef.current = automationActive;
  const hwRef = useRef(hardware);
  hwRef.current = hardware;
  const historyRef = useRef(history);
  historyRef.current = history;
  const tickRef = useRef(0);
  
  const modelRef = useRef<tf.Sequential | null>(null);

  // Initialize Genuine TF.js Neural Network
  useEffect(() => {
    const initModel = async () => {
      // Small Multi-Layer Perceptron (Surrogate Model)
      const model = tf.sequential();
      // Input: 4 Environmental states + 3 Hardware = 7 Features
      model.add(tf.layers.dense({inputShape: [7], units: 24, activation: 'relu'}));
      model.add(tf.layers.dropout({rate: 0.2}));
      model.add(tf.layers.dense({units: 12, activation: 'relu'}));
      model.add(tf.layers.dropout({rate: 0.2}));
      model.add(tf.layers.dense({units: 1})); // Output: predicted BO Score
      model.compile({optimizer: tf.train.adam(0.02), loss: 'meanSquaredError'});
      modelRef.current = model;
    };
    initModel();
  }, []);

  // Main Event Loop (Simulation + Real ML Training + Inference)
  useEffect(() => {
    let isRunning = true;
    
    // Default initial
    historyRef.current = [{
      iter: 0, algae: 5000, ph: 7.2, temp: 20.0, cond: 30.0, power: 0, score: 0, inactRate: 0, hw: hwRef.current
    }];
    setHistory([...historyRef.current]);

    const loop = async () => {
      while (isRunning) {
        tickRef.current += 1;
        const prev = historyRef.current;
        const last = prev[prev.length - 1];
        const currentHw = hwRef.current;

        // 1. PHYSICAL SIMULATION OF SENSORS (Generating true reward data)
        const plasmaIntensity = (currentHw.capacitorLevel * 100) / Math.max(0.5, (currentHw.ballGap * 0.6 + currentHw.tungstenGap * 8));
        const actualPower = plasmaIntensity;

        const decayFactor = 1 - Math.min(0.08, actualPower * 0.002);
        const nextAlgae = Math.max(50, last.algae * decayFactor + (Math.random() * 40 - 20));
        
        const inactRate = Math.max(0, ((5000 - nextAlgae) / 5000) * 100);
        const energyPenalty = actualPower * 0.1; 
        
        // Target truth to learn
        const boScore = Math.max(0, (inactRate * 0.8) - energyPenalty + (Math.random() * 2));

        const nextPoint = {
          iter: tickRef.current,
          algae: nextAlgae,
          ph: Math.max(6.5, last.ph - (actualPower * 0.0008) + (Math.random() * 0.02 - 0.01)),
          temp: Math.min(35, last.temp + (actualPower * 0.05) - ((last.temp - 20) * 0.1)),
          cond: Math.min(150, last.cond + (actualPower * 0.08)),
          power: actualPower,
          inactRate: inactRate,
          score: boScore,
          hw: {...currentHw} // Store the hardware that produced this score
        };

        historyRef.current = [...historyRef.current.slice(-40), nextPoint]; // Keep last 40 ticks
        setHistory(historyRef.current);

        // Give React a moment to render UI
        await new Promise(r => setTimeout(r, 200));

        // 2. TRUE DNN ONLINE LEARNING & OPTIMIZATION (Takes over if Automated)
        if (autoRef.current && modelRef.current && historyRef.current.length > 5) {
            
            // === STEP 2.A: TRAIN THE SURROGATE MODEL ===
            const dataToTrain = historyRef.current;
            const xs = tf.tensor2d(dataToTrain.map(d => [
               d.algae/5000, d.ph/14, d.temp/50, d.cond/150, 
               d.hw.ballGap/20, d.hw.tungstenGap/5, d.hw.capacitorLevel/5
            ]));
            const ys = tf.tensor2d(dataToTrain.map(d => [d.score/100])); // Normalize score
            
            const hInfo = await modelRef.current.fit(xs, ys, {epochs: 2, verbose: 0});
            setLossData(prev => [...prev.slice(-30), { epoch: prev.length, loss: hInfo.history.loss[0] as number }]);
            addLog(`模型增量迭代 - Epoch MSE: ${(hInfo.history.loss[0] as number).toExponential(3)}`, 'system');
            
            xs.dispose();
            ys.dispose();

            // === STEP 2.B: RUN SURROGATE SEARCH (Bayesian Optimization via MC Dropout) ===
            // Generate 300 possible hardware candidates for the CURRENT state
            tf.tidy(() => {
               const NUM_PASSES = 10; // MC Dropout passes for uncertainty estimation
               const KAPPA = 2.0;    // Upper Confidence Bound (UCB) exploration parameter
            
               const candidates: any[] = [];
               for(let i=0; i<300; i++) {
                 candidates.push({
                   ballGap: 2.0 + Math.random() * 18,
                   tungstenGap: 0.1 + Math.random() * 4.9,
                   capacitorLevel: Math.floor(1 + Math.random() * 5)
                 });
               }
               
               const xsPred = tf.tensor2d(candidates.map(c => [
                  nextPoint.algae/5000, nextPoint.ph/14, nextPoint.temp/50, nextPoint.cond/150, 
                  c.ballGap/20, c.tungstenGap/5, c.capacitorLevel/5
               ]));
               
               // 1. Multiple stochastic forward passes (training: true enables Dropout)
               const passResults: tf.Tensor[] = [];
               for(let i=0; i<NUM_PASSES; i++) {
                  passResults.push(modelRef.current!.apply(xsPred, {training: true}) as tf.Tensor);
               }
               
               // 2. Calculate UCB: Mean + Kappa * StdDev
               const stacked = tf.stack(passResults); // shape: [NUM_PASSES, NUM_CANDIDATES, 1]
               const moments = tf.moments(stacked, 0); 
               const means = moments.mean.squeeze(); // shape: [NUM_CANDIDATES]
               const variances = moments.variance.squeeze();
               const stdDevs = variances.sqrt();
               
               const ucb = means.add(stdDevs.mul(KAPPA));
               const ucbValues = ucb.dataSync();
               
               // 3. Find candidate with maximum UCB score
               let maxIdx = 0, maxUCB = -Infinity;
               for(let i=0; i<ucbValues.length; i++) {
                  if(ucbValues[i] > maxUCB) {
                     maxUCB = ucbValues[i];
                     maxIdx = i;
                  }
               }
               
               const chosenHw = candidates[maxIdx];

               // Populate Scatter plot data
               const plotData = Array.from(ucbValues).map((val, idx) => ({
                 b: candidates[idx].ballGap,
                 t: candidates[idx].tungstenGap,
                 s: val
               })).filter((_, i) => i % 5 === 0);
               setScatterData(plotData);

               const formattedHw = {
                 ballGap: Number(chosenHw.ballGap.toFixed(1)),
                 tungstenGap: Number(chosenHw.tungstenGap.toFixed(2)),
                 capacitorLevel: chosenHw.capacitorLevel
               };
               
               hwRef.current = formattedHw;
               setHardware(formattedHw);
               addLog(`BO: 解析300维侯选空间，锁定上限置信界 (UCB: ${maxUCB.toFixed(2)})。已分发硬件指令。`, 'success');
            });
        } else if (autoRef.current && historyRef.current.length <= 5) {
            if (tickRef.current % 2 === 0) addLog(`缓冲样本采集中... (${historyRef.current.length}/5)`, 'info');
        } else if (!autoRef.current && tickRef.current % 10 === 0) {
            addLog(`物理系统开环稳压运行: 本侧藻密度观测 ${Math.round(nextAlgae)} cells/mL`, 'info');
        }

        // Wait for next cycle
        await new Promise(r => setTimeout(r, 1300)); 
      }
    };

    loop();

    return () => { isRunning = false; };
  }, []);

  const handleManualHardwareChange = (key: string, val: number) => {
    if (automationActive) return; // Block manual if AI is active
    setHardware({ ...hardware, [key]: val });
  };

  const currentData = history.length > 0 ? history[history.length - 1] : null;

  return (
    <div className="h-screen w-full bg-[#080808] text-[#e0e0e0] font-sans flex flex-col selection:bg-cyan-500/30 overflow-hidden">
      
      {/* HEADER */}
      <header className="h-[64px] border-b border-[#222] flex items-center px-6 justify-between bg-[#080808] shrink-0 z-50">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 rounded shrink-0 bg-[#111] border border-[#333] flex items-center justify-center text-white">
            <Zap size={16} />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-[#f5f5f5] flex items-center gap-3">
              SPA-BO/DNN 控制台
            </h1>
          </div>
        </div>
        <div className="flex gap-4 items-center">
          <div className="flex items-center gap-2 px-3 py-1 bg-[#111] border border-[#222] rounded">
            <div className={`w-2 h-2 rounded-none ${automationActive ? 'bg-emerald-500' : 'bg-red-500'}`}></div>
            <span className="text-[10px] font-mono text-[#888] uppercase">MCU {automationActive ? '同步中' : '离线 (Offline)'}</span>
          </div>
          <button 
            onClick={() => setAutomationActive(!automationActive)} 
            className={`flex items-center gap-2 px-5 py-2 rounded text-xs font-mono font-semibold transition-all duration-300 border ${
              automationActive 
                ? 'bg-[#1a2e1f] border-emerald-800 text-emerald-400' 
                : 'bg-[#111] border-[#333] text-[#aaa] hover:bg-[#1a1a1a] hover:text-white'
            }`}
          >
            <Bot size={14}/> 
            {automationActive ? '停止学习 (Stop Learning)' : '开始学习 (Start Learning)'}
          </button>
        </div>
      </header>

      {/* OVERALL GRID */}
      <main className="flex-1 grid grid-cols-12 gap-6 p-6 overflow-hidden relative z-10">
        
        {/* ================= LEFT PANEL: SENSORS (COL 3) ================= */}
        <div className="col-span-3 flex flex-col gap-4 min-h-0">
          <div className="text-xs font-bold text-zinc-400 uppercase tracking-widest flex items-center gap-2 px-1 shrink-0">
            <Activity size={14}/> 实时环境传感矩阵 (Sensors)
          </div>
          
          <div className="flex-1 grid grid-rows-4 gap-4">
            {/* SENSOR 1: Algae */}
            <div className="bg-[#111] border border-[#222] rounded flex flex-col relative overflow-hidden p-3 hover:border-[#444] transition-colors">
              <div className="flex justify-between items-start z-10">
                <div className="flex items-center gap-2 text-[#aaa] text-xs font-semibold tracking-wide uppercase"><Droplets size={14} className="text-cyan-500"/> 蓝藻密度 (Algae)</div>
                <div className="text-[10px] text-[#666] font-mono">cells/mL</div>
              </div>
              <div className="text-2xl font-mono tracking-tight text-[#f5f5f5] mt-1 z-10 tabular-nums">{currentData?.algae.toFixed(0)}</div>
              <div className="absolute -bottom-2 -left-2 -right-2 h-20 opacity-80 pointer-events-none">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history}>
                    <defs>
                      <linearGradient id="colorAlgae" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#22d3ee" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <YAxis domain={['auto', 'auto']} hide/>
                    <Area type="step" dataKey="algae" stroke="#22d3ee" fillOpacity={1} fill="url(#colorAlgae)" strokeWidth={1} isAnimationActive={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* SENSOR 2: pH */}
            <div className="bg-[#111] border border-[#222] rounded flex flex-col relative overflow-hidden p-3 hover:border-[#444] transition-colors">
              <div className="flex justify-between items-start z-10">
                <div className="flex items-center gap-2 text-[#aaa] text-xs font-semibold tracking-wide uppercase"><ShieldAlert size={14} className="text-emerald-500"/> 酸碱度 (pH)</div>
              </div>
              <div className="text-2xl font-mono tracking-tight text-[#f5f5f5] mt-1 z-10 tabular-nums">{currentData?.ph.toFixed(2)}</div>
              <div className="absolute -bottom-2 -left-2 -right-2 h-20 opacity-80 pointer-events-none">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history}>
                    <defs>
                      <linearGradient id="colorPh" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#34d399" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#34d399" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <YAxis domain={[5.5, 8.5]} hide/>
                    <Area type="step" dataKey="ph" stroke="#34d399" fill="url(#colorPh)" strokeWidth={1} isAnimationActive={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* SENSOR 3: Temperature */}
            <div className="bg-[#111] border border-[#222] rounded flex flex-col relative overflow-hidden p-3 hover:border-[#444] transition-colors">
              <div className="flex justify-between items-start z-10">
                <div className="flex items-center gap-2 text-[#aaa] text-xs font-semibold tracking-wide uppercase"><Thermometer size={14} className="text-rose-500"/> 水体温度 (Temp)</div>
                <div className="text-[10px] text-[#666] font-mono">°C</div>
              </div>
              <div className="text-2xl font-mono tracking-tight text-[#f5f5f5] mt-1 z-10 tabular-nums">{currentData?.temp.toFixed(1)}</div>
              <div className="absolute -bottom-2 -left-2 -right-2 h-20 opacity-80 pointer-events-none">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history}>
                    <defs>
                      <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#fb7185" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#fb7185" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <YAxis domain={[15, 45]} hide/>
                    <Area type="step" dataKey="temp" stroke="#fb7185" fill="url(#colorTemp)" strokeWidth={1} isAnimationActive={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* SENSOR 4: Conductivity */}
            <div className="bg-[#111] border border-[#222] rounded flex flex-col relative overflow-hidden p-3 hover:border-[#444] transition-colors">
              <div className="flex justify-between items-start z-10">
                <div className="flex items-center gap-2 text-[#aaa] text-xs font-semibold tracking-wide uppercase"><ZapIcon size={14} className="text-amber-500"/> 反应电导率 (Cond)</div>
                <div className="text-[10px] text-[#666] font-mono">mS/cm</div>
              </div>
              <div className="text-2xl font-mono tracking-tight text-[#f5f5f5] mt-1 z-10 tabular-nums">{currentData?.cond.toFixed(1)}</div>
              <div className="absolute -bottom-2 -left-2 -right-2 h-20 opacity-80 pointer-events-none">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history}>
                    <defs>
                      <linearGradient id="colorCond" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#fbbf24" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <YAxis domain={['dataMin - 10', 'dataMax + 10']} hide/>
                    <Area type="step" dataKey="cond" stroke="#fbbf24" fill="url(#colorCond)" strokeWidth={1} isAnimationActive={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>

        {/* ================= CENTER PANEL: OPTIMIZATION HUB (COL 6) ================= */}
        <div className="col-span-6 flex flex-col gap-4 min-h-0">
           <div className="text-xs font-bold text-[#888] uppercase tracking-widest flex items-center justify-between px-1 shrink-0">
              <div className="flex items-center gap-2"><Cpu size={14} className="text-indigo-400"/> 决策拓扑与性能观测核心 (BO-DNN Core)</div>
              <div className="text-[10px] font-mono text-[#555] bg-[#111] px-2 py-1 rounded border border-[#222]">TICK: <span className="text-[#ddd]">{currentData?.iter || 0}</span></div>
           </div>

           {/* KPI Cards */}
           <div className="grid grid-cols-3 gap-4">
              <div className="bg-[#111] border border-[#222] rounded p-4 flex flex-col justify-between">
                 <div className="text-[10px] text-[#666] font-mono tracking-widest uppercase mb-1">Inactivation Rate (%)</div>
                 <div className="flex items-baseline gap-1.5">
                   <div className="text-3xl font-bold font-mono tracking-tighter text-[#f5f5f5] tabular-nums">{currentData?.inactRate.toFixed(1)}</div>
                   <div className="text-xs font-mono text-emerald-500">%</div>
                 </div>
              </div>
              <div className="bg-[#111] border border-[#222] rounded p-4 flex flex-col justify-between">
                 <div className="text-[10px] text-[#666] font-mono tracking-widest uppercase mb-1">Energy (J/L)</div>
                 <div className="flex items-baseline gap-1.5">
                   <div className="text-3xl font-bold font-mono tracking-tighter text-[#f5f5f5] tabular-nums">{currentData?.power.toFixed(0)}</div>
                   <div className="text-xs font-mono text-cyan-500">J/L</div>
                 </div>
              </div>
              <div className="bg-[#111] border border-[#222] rounded p-4 flex flex-col justify-between relative overflow-hidden">
                 <div className="text-[10px] text-[#666] font-mono tracking-widest uppercase mb-1">BO Target Score</div>
                 <div className="flex items-baseline gap-2 relative z-10">
                   <div className="text-3xl font-bold font-mono tracking-tighter text-[#f5f5f5] tabular-nums">{currentData?.score.toFixed(1)}</div>
                   <div className="text-xs font-mono text-indigo-400">[OBJ]</div>
                 </div>
              </div>
           </div>

           {/* Charts Container */}
           <div className="flex-1 flex flex-col gap-4">
              
              {/* Main Double Axis Chart */}
              <div className="bg-[#111] border border-[#222] rounded p-5 relative flex flex-col min-h-[220px]">
                 <div className="flex justify-between items-end mb-4">
                    <div>
                      <h3 className="text-xs font-semibold text-[#ccc] tracking-wide">态势演化矩阵 / Evaluation Topology</h3>
                    </div>
                 </div>

                 <div className="flex-1 -ml-4">
                   <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={history} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                         <CartesianGrid strokeDasharray="2 2" stroke="#222" vertical={false} />
                         <XAxis dataKey="iter" stroke="#444" tick={{fontSize: 9, fill: '#666', fontFamily: 'var(--font-mono)'}} tickMargin={8} axisLine={false} tickLine={false} />
                         <YAxis yAxisId="left" stroke="#444" tick={{fontSize: 9, fill: '#666', fontFamily: 'var(--font-mono)'}} tickFormatter={(val) => `${val}%`} axisLine={false} tickLine={false} domain={[0, 100]} />
                         <YAxis yAxisId="right" orientation="right" stroke="#444" tick={{fontSize: 9, fill: '#666', fontFamily: 'var(--font-mono)'}} axisLine={false} tickLine={false} domain={[0, 'auto']} />
                         <Tooltip 
                           contentStyle={{backgroundColor: '#161616', borderColor: '#333', borderRadius: '4px', color: '#fff'}} 
                           itemStyle={{fontSize: '11px', fontFamily: 'var(--font-mono)'}}
                           labelStyle={{color: '#888', marginBottom: '6px', fontSize: '10px', textTransform: 'uppercase'}}
                         />
                         <Legend wrapperStyle={{fontSize: '10px', paddingTop: '10px', color: '#888', fontFamily: 'var(--font-mono)'}} iconType="rect" iconSize={8}/>

                         <Bar yAxisId="right" dataKey="power" name="Energy (J/L)" fill="#22d3ee" fillOpacity={0.2} isAnimationActive={false} barSize={4} />
                         <Line yAxisId="left" type="step" dataKey="inactRate" name="Inact Rate (%)" stroke="#34d399" strokeWidth={1} dot={false} isAnimationActive={false}/>
                         <Line yAxisId="left" type="step" dataKey="score" name="Obj Score" stroke="#818cf8" strokeWidth={1.5} dot={false} isAnimationActive={false}/>
                      </ComposedChart>
                   </ResponsiveContainer>
                 </div>
              </div>

              {/* Lower Double Charts Row */}
              <div className="grid grid-cols-2 gap-4 flex-1 min-h-[160px]">
                {/* Real DNN Loss Chart */}
                <div className="bg-[#111] border border-[#222] rounded p-4 flex flex-col relative overflow-hidden">
                   <div className="flex justify-between items-start mb-3 z-10">
                      <div>
                        <h3 className="text-xs font-semibold text-[#ccc] flex items-center gap-2">
                          代理网络损失 (Loss)
                        </h3>
                        <p className="text-[9px] text-[#555] mt-1 font-mono uppercase tracking-widest">MSE / ADAM</p>
                      </div>
                      <div className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${automationActive ? 'bg-emerald-900/30 text-emerald-500' : 'bg-[#1a1a1a] text-[#555]'}`}>
                        {automationActive ? "FIT()" : "IDLE"}
                      </div>
                   </div>

                   <div className="flex-1 w-full z-10 -ml-4">
                     {lossData.length > 0 ? (
                       <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={lossData} margin={{ top: 5, right: 10, left: -5, bottom: 0 }}>
                             <CartesianGrid strokeDasharray="2 2" stroke="#222" vertical={false} />
                             <XAxis dataKey="epoch" stroke="#444" tick={{fontSize: 9, fontFamily: 'var(--font-mono)'}} axisLine={false} tickLine={false} />
                             <YAxis stroke="#444" tick={{fontSize: 9, fontFamily: 'var(--font-mono)'}} tickFormatter={(val) => val.toExponential(1)} axisLine={false} tickLine={false} />
                             <Tooltip 
                               contentStyle={{backgroundColor: '#161616', borderColor: '#333', borderRadius: '4px', color: '#fff'}} 
                               itemStyle={{color: '#f43f5e', fontSize: '11px', fontFamily: 'var(--font-mono)'}}
                               labelStyle={{color: '#888', fontSize: '9px', fontFamily: 'var(--font-mono)'}}
                               formatter={(value: number) => [value.toExponential(4), 'Loss']}
                             />
                             <Line type="stepAfter" dataKey="loss" stroke="#f43f5e" strokeWidth={1} dot={false} isAnimationActive={false} />
                          </LineChart>
                       </ResponsiveContainer>
                     ) : (
                       <div className="w-full h-full flex items-center justify-center text-[10px] text-[#444] font-mono">
                         等待数据 (Waiting)
                       </div>
                     )}
                   </div>
                </div>

                {/* BO Candidate Scatter Plot */}
                <div className="bg-[#111] border border-[#222] rounded p-4 flex flex-col relative overflow-hidden">
                   <div className="flex justify-between items-start mb-3 z-10">
                      <div>
                        <h3 className="text-xs font-semibold text-[#ccc] flex items-center gap-2">
                          UCB 置信界 (UCB Horizon)
                        </h3>
                        <p className="text-[9px] text-[#555] mt-1 font-mono uppercase tracking-widest">候选空间 (Candidates)</p>
                      </div>
                      <div className={`text-[9px] px-1.5 py-0.5 rounded font-mono bg-indigo-900/30 text-indigo-400`}>
                        {scatterData.length} 点
                      </div>
                   </div>

                   <div className="flex-1 w-full z-10 -ml-4">
                     {scatterData.length > 0 ? (
                       <ResponsiveContainer width="100%" height="100%">
                          <ScatterChart margin={{ top: 5, right: 10, bottom: 5, left: -5 }}>
                            <CartesianGrid strokeDasharray="2 2" stroke="#222" />
                            <XAxis type="number" dataKey="b" name="Ball Gap" stroke="#444" tick={{fontSize: 9}}  domain={[0, 20]} />
                            <YAxis type="number" dataKey="t" name="Tungsten Gap" stroke="#444" tick={{fontSize: 9}} domain={[0, 5]} />
                            <ZAxis type="number" dataKey="s" range={[10, 80]} name="Score" />
                            <Tooltip cursor={{strokeDasharray: '3 3', stroke: '#555'}} 
                              contentStyle={{backgroundColor: '#161616', borderColor: '#333', borderRadius: '4px', color: '#fff'}}
                              itemStyle={{fontSize: '11px', fontFamily: 'var(--font-mono)'}}
                            />
                            <Scatter name="Candidates" data={scatterData} isAnimationActive={false}>
                              {scatterData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.s > 50 ? '#34d399' : entry.s > 20 ? '#818cf8' : '#fb7185'} />
                              ))}
                            </Scatter>
                          </ScatterChart>
                       </ResponsiveContainer>
                     ) : (
                       <div className="w-full h-full flex items-center justify-center text-[10px] text-[#444] font-mono">
                         等待数据 (Waiting)
                       </div>
                     )}
                   </div>
                </div>
              </div>

           </div>
        </div>

        {/* ================= RIGHT PANEL: HARDWARE & TERMINAL (COL 3) ================= */}
        <div className="col-span-3 flex flex-col gap-4 min-h-0">
           <div className="text-xs font-bold text-[#888] uppercase tracking-widest flex items-center justify-between px-1 shrink-0">
              <div className="flex items-center gap-2"><Settings2 size={14} className={automationActive ? "text-indigo-500" : "text-amber-500"}/> 控制参数链路 (Hardware)</div>
              {automationActive && <div className="text-[9px] bg-indigo-900/30 text-indigo-400 px-2 py-0.5 rounded border border-indigo-900/50 font-mono">AI CONTROL</div>}
           </div>

           <div className={`flex flex-col gap-3 bg-[#111] border border-[#222] rounded p-4 relative overflow-hidden shrink-0 ${automationActive ? 'opacity-60 pointer-events-none' : ''}`}>
              
              {automationActive && (
                  <div className="absolute top-0 left-0 w-full h-1 bg-indigo-500 z-50"></div>
              )}

              <div className="space-y-4">
                {/* Control 1 */}
                <div className="bg-[#161616] border border-[#222] rounded p-3 group">
                   <div className="flex justify-between items-center mb-3">
                      <div>
                        <div className="text-[11px] font-semibold text-[#ccc]">球隙激发距离 (Spark Gap)</div>
                      </div>
                      <div className="text-sm font-mono text-cyan-400 tabular-nums">{hardware.ballGap.toFixed(1)} <span className="text-[9px] text-[#666] font-sans">mm</span></div>
                   </div>
                   <input 
                      type="range" min={2.0} max={20.0} step={0.1}
                      value={hardware.ballGap}
                      onChange={(e) => handleManualHardwareChange('ballGap', parseFloat(e.target.value))}
                      className="w-full h-1 bg-[#222] rounded appearance-none cursor-pointer accent-cyan-500 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-sm [&::-webkit-slider-thumb]:bg-cyan-500"
                   />
                </div>

                {/* Control 2 */}
                <div className="bg-[#161616] border border-[#222] rounded p-3 group">
                   <div className="flex justify-between items-center mb-3">
                      <div>
                        <div className="text-[11px] font-semibold text-[#ccc]">钨针辐射间距 (Tungsten Gap)</div>
                      </div>
                      <div className="text-sm font-mono text-rose-400 tabular-nums">{hardware.tungstenGap.toFixed(2)} <span className="text-[9px] text-[#666] font-sans">mm</span></div>
                   </div>
                   <input 
                      type="range" min={0.1} max={5.0} step={0.05}
                      value={hardware.tungstenGap}
                      onChange={(e) => handleManualHardwareChange('tungstenGap', parseFloat(e.target.value))}
                      className="w-full h-1 bg-[#222] rounded appearance-none cursor-pointer accent-rose-500 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-sm [&::-webkit-slider-thumb]:bg-rose-500"
                   />
                </div>

                {/* Control 3 */}
                <div className="bg-[#161616] border border-[#222] rounded p-3 group">
                   <div className="flex justify-between items-center mb-3">
                      <div>
                        <div className="text-[11px] font-semibold text-[#ccc]">电容脉冲能级 (Capacitor Tier)</div>
                      </div>
                      <div className="text-[10px] font-mono text-amber-500 bg-amber-900/20 px-2 py-0.5 rounded border border-amber-900/30">Lv {hardware.capacitorLevel}</div>
                   </div>
                   <input 
                      type="range" min={1} max={5} step={1}
                      value={hardware.capacitorLevel}
                      onChange={(e) => handleManualHardwareChange('capacitorLevel', parseFloat(e.target.value))}
                      className="w-full h-1 bg-[#222] rounded appearance-none cursor-pointer accent-amber-500 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-sm [&::-webkit-slider-thumb]:bg-amber-500"
                   />
                </div>
              </div>
           </div>

           {/* LOG TERMINAL */}
           <div className="flex-1 bg-[#111] border border-[#222] rounded flex flex-col overflow-hidden min-h-[200px]">
             <div className="bg-[#161616] border-b border-[#222] px-3 py-2 flex justify-between items-center shrink-0">
               <div className="text-[10px] font-bold text-[#888] flex items-center gap-2 font-mono uppercase tracking-widest">
                 <TerminalIcon size={12} className="text-[#aaa]" />
                 系统日志 (System Logs)
               </div>
               <div className="flex justify-start gap-1">
                 <div className="w-2 h-2 rounded-full bg-[#333]"></div>
                 <div className="w-2 h-2 rounded-full bg-[#333]"></div>
                 <div className="w-2 h-2 rounded-full bg-[#333]"></div>
               </div>
             </div>
             <div className="flex-1 overflow-y-auto p-3 flex flex-col-reverse gap-1.5 custom-scrollbar">
               {[...logs].reverse().map((log, i) => (
                 <div key={i} className="text-[10px] font-mono leading-relaxed break-words">
                   <span className="text-[#555] mr-2">[{log.time}]</span>
                   <span className={
                     log.type === 'error' ? 'text-rose-500' :
                     log.type === 'warn' ? 'text-amber-500' :
                     log.type === 'success' ? 'text-emerald-500' :
                     log.type === 'system' ? 'text-indigo-400' :
                     'text-[#aaa]'
                   }>
                     {log.msg}
                   </span>
                 </div>
               ))}
             </div>
           </div>
        </div>

      </main>
    </div>
  );
}
