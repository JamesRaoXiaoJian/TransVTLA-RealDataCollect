import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Camera,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  GitBranch,
  Layers3,
  Radar,
  SlidersHorizontal,
} from "lucide-react";
import "./style.css";

const pipeline = [
  { label: "采集", detail: "外部/腕部 RGB-D + 触觉 + 机械臂 + 夹爪", icon: Camera },
  { label: "审计", detail: "模态完整性 / 频率 / 时间戳 / 同步偏移", icon: Radar },
  { label: "清洗", detail: "20 通道归一 / CH58 插值 / 去重 / 对齐重建", icon: SlidersHorizontal },
  { label: "划分", detail: "硬件视角划分 + 任务视图 + 软链接清单", icon: GitBranch },
  { label: "导出", detail: "RGB / RGB-D / 对齐训练集 / 可转 RLDS", icon: Database },
];

const overviewKpis = [
  { value: "1,000", label: "真机轨迹" },
  { value: "10,000", label: "仿真轨迹" },
  { value: "4", label: "任务视图" },
  { value: "1:10", label: "真机/仿真比例" },
];

const taskAllocation = [
  { name: "T1 简单放入盒子", real: 400, sim: 4000, color: "#2f80ed" },
  { name: "T2 复杂盒子/烧杯场景", real: 220, sim: 2200, color: "#10a37f" },
  { name: "T3 烧杯放置三脚架", real: 180, sim: 1800, color: "#9c36b5" },
  { name: "T4 试管插入试管架", real: 200, sim: 2000, color: "#e67700" },
];

const modalityRows = [
  ["真机采集", "1,000", "RGB-D / 触觉 / 机械臂 / 夹爪", "按任务均衡"],
  ["仿真采集", "10,000", "同一数据结构 + 同一任务标签", "与真机域对齐"],
  ["视觉同步", "外部 + 腕部", "RGB-D 深度对齐", "可直接回放"],
  ["状态同步", "触觉 + 机械臂", "时间戳对齐", "可用于训练"],
];

const cleanActions = [
  ["真机配额", "4 个任务共 1,000 条轨迹"],
  ["仿真配额", "10,000 条轨迹，是真机的 10 倍"],
  ["数据结构", "相机 / 触觉 / 机械臂状态保持一致"],
  ["对齐方式", "共享时间戳逻辑和任务标签"],
];

const tasks = [
  {
    code: "T1",
    title: "简单放入盒子",
    cn: "目标物体放入木盒",
    sessions: 400,
    sim: 4000,
    frames: "真机 400 / 仿真 4,000",
    sample: "session_20260615_203217",
    split: "dual_realsense_repositioned",
    frame: "0194",
    rows: { aligned: 387, pressure: 2555, robot: "90.8 Hz", gripper: "82.6 Hz" },
    images: ["/assets/task1_world.jpg", "/assets/task1_wrist.jpg"],
    accent: "#2f80ed",
  },
  {
    code: "T2",
    title: "复杂盒子/烧杯",
    cn: "盒子与烧杯三脚架混合场景",
    sessions: 220,
    sim: 2200,
    frames: "真机 220 / 仿真 2,200",
    sample: "session_20260616_143430",
    split: "dual_realsense_repositioned",
    frame: "0149",
    rows: { aligned: 297, pressure: 2077, robot: "96.1 Hz", gripper: "87.4 Hz" },
    images: ["/assets/task2_world.jpg", "/assets/task2_wrist.jpg"],
    accent: "#10a37f",
  },
  {
    code: "T3",
    title: "烧杯放置三脚架",
    cn: "简单烧杯与三脚架场景",
    sessions: 180,
    sim: 1800,
    frames: "真机 180 / 仿真 1,800",
    sample: "session_20260615_210748",
    split: "dual_realsense_repositioned",
    frame: "0221",
    rows: { aligned: 441, pressure: 2913, robot: "88.4 Hz", gripper: "79.6 Hz" },
    images: ["/assets/task3_world.jpg", "/assets/task3_wrist.jpg"],
    accent: "#9c36b5",
  },
  {
    code: "T4",
    title: "试管插入试管架",
    cn: "多孔试管架插入任务",
    sessions: 200,
    sim: 2000,
    frames: "真机 200 / 仿真 2,000",
    sample: "session_20260615_214044",
    split: "dual_realsense_repositioned",
    frame: "0251",
    rows: { aligned: 501, pressure: 3285, robot: "88.0 Hz", gripper: "77.3 Hz" },
    images: ["/assets/task4_world.jpg", "/assets/task4_wrist.jpg"],
    accent: "#e67700",
  },
];

function App() {
  const isExport = new URLSearchParams(window.location.search).get("export") === "1";
  const [slide, setSlide] = useState(() => {
    const requested = Number(new URLSearchParams(window.location.search).get("slide") || "1");
    return Math.max(0, Math.min(1, requested - 1));
  });
  const slides = useMemo(() => [<StrategySlide />, <TaskViewSlide />], []);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "ArrowRight") setSlide((value) => Math.min(value + 1, slides.length - 1));
      if (event.key === "ArrowLeft") setSlide((value) => Math.max(value - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [slides.length]);

  return (
    <main className={`deck${isExport ? " export-mode" : ""}`}>
      <div className="slide-shell">{slides[slide]}</div>
      {!isExport && (
        <div className="deck-nav" aria-label="slide navigation">
          <button type="button" onClick={() => setSlide(Math.max(slide - 1, 0))} aria-label="上一页">
            <ChevronLeft size={18} />
          </button>
          <span>{slide + 1} / {slides.length}</span>
          <button type="button" onClick={() => setSlide(Math.min(slide + 1, slides.length - 1))} aria-label="下一页">
            <ChevronRight size={18} />
          </button>
        </div>
      )}
    </main>
  );
}

function StrategySlide() {
  return (
    <section className="slide strategy-slide">
      <SlideHeader eyebrow="TransVTLA 数据工程" title="数据工程策略与当前概览" meta="真机 1,000 / 仿真 10,000" />
      <div className="two-column">
        <div className="left-column">
          <div className="section-title">
            <Layers3 size={18} />
            <span>工程策略</span>
          </div>
          <div className="pipeline">
            {pipeline.map((item, index) => {
              const Icon = item.icon;
              return (
                <div className="pipeline-row" key={item.label}>
                  <div className="step-index">{String(index + 1).padStart(2, "0")}</div>
                  <div className="step-icon"><Icon size={18} /></div>
                  <div className="step-copy">
                    <strong>{item.label}</strong>
                    <span>{item.detail}</span>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="clean-grid">
            {cleanActions.map(([label, value]) => (
              <div className="clean-item" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
        </div>
        <div className="right-column">
          <div className="section-title">
            <Database size={18} />
            <span>当前数据概览</span>
          </div>
          <div className="kpi-grid">
            {overviewKpis.map((item) => (
              <div className="kpi" key={item.label}>
                <strong>{item.value}</strong>
                <span>{item.label}</span>
              </div>
            ))}
          </div>
          <div className="split-bars">
            {taskAllocation.map((item) => (
              <div className="split-row" key={item.name}>
                <div className="split-label">
                  <span>{item.name}</span>
                  <strong>{item.real} 真机 + {item.sim} 仿真</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${(item.real / 1000) * 100}%`, background: item.color }} />
                </div>
                <span className="frame-count">共 {item.real + item.sim} 条</span>
              </div>
            ))}
          </div>
          <table className="modality-table">
            <thead>
              <tr>
                <th>数据轨道</th>
                <th>数量</th>
                <th>覆盖内容</th>
                <th>对齐状态</th>
              </tr>
            </thead>
            <tbody>
              {modalityRows.map((row) => (
                <tr key={row[0]}>
                  {row.map((cell) => <td key={cell}>{cell}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function TaskViewSlide() {
  return (
    <section className="slide task-slide">
      <SlideHeader eyebrow="任务视图" title="四任务采集/回放视图矩阵" meta="真机 1,000 / 仿真 10,000" />
      <div className="task-grid">
        {tasks.map((task) => <TaskConsole task={task} key={task.code} />)}
      </div>
    </section>
  );
}

function TaskConsole({ task }) {
  return (
    <article className="task-console" style={{ "--accent": task.accent }}>
      <div className="console-head">
        <div>
          <div className="task-code">{task.code}</div>
          <h2>{task.title}</h2>
          <p>{task.cn}</p>
        </div>
        <div className="task-counts">
          <strong>{task.sessions}</strong>
          <span>真机</span>
          <em>{task.sim} 仿真</em>
        </div>
      </div>
      <div className="viewer-strip">
        <CameraView label="外部 RGB" src={task.images[0]} />
        <CameraView label="腕部 RGB" src={task.images[1]} />
      </div>
      <div className="playback-row">
        <span>{task.sample}</span>
        <strong>帧 {task.frame} | 仿真对齐</strong>
      </div>
      <div className="telemetry">
        <Metric icon={Camera} label="对齐帧" value={task.rows.aligned} />
        <Metric icon={Activity} label="触觉行" value={task.rows.pressure} />
        <Metric icon={Cpu} label="机械臂" value={task.rows.robot} />
        <Metric icon={SlidersHorizontal} label="夹爪" value={task.rows.gripper} />
      </div>
    </article>
  );
}

function CameraView({ label, src }) {
  return (
    <div className="camera-view">
      <img src={src} alt={label} />
      <span>{label}</span>
    </div>
  );
}

function Metric({ icon: Icon, label, value }) {
  return (
    <div className="metric">
      <Icon size={14} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SlideHeader({ eyebrow, title, meta }) {
  return (
    <header className="slide-header">
      <div>
        <span>{eyebrow}</span>
        <h1>{title}</h1>
      </div>
      <strong>{meta}</strong>
    </header>
  );
}

createRoot(document.getElementById("root")).render(<App />);
