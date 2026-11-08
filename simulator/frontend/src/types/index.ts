export interface Detection {
  bbox: [number, number, number, number];
  confidence: number;
  class_id: number;
  species_name: string;
}

export interface Timing {
  preprocess_ms: number;
  inference_ms: number;
  postprocess_ms: number;
  total_ms: number;
}

export interface ModelResult {
  model_name: string;
  is_mock: boolean;
  detections: Detection[];
  detection_count: number;
  image_size: [number, number];
  timing: Timing;
}

export interface Showdown {
  winner: string;
  faster_model: string;
  delta_ms: number;
  speedup_pct: number;
  baseline_ms: number;
  improved_ms: number;
  baseline_fps: number;
  improved_fps: number;
}

export interface DualPredictionResponse {
  baseline: ModelResult;
  improved: ModelResult;
  showdown: Showdown;
}

export interface MacroRow {
  metric: string;
  baseline: number;
  improved: number;
  delta: number;
  unit: string;
  higher_is_better: boolean;
}

export interface SpeciesItem {
  class_id: number;
  species: string;
  order: string;
  baseline_map50: number;
  improved_map50: number;
  delta_map50: number;
  improved_p: number;
  improved_r: number;
  improved_map50_95: number;
}

export interface ArchitectureSpec {
  component: string;
  baseline: string;
  improved: string;
}

export interface StatisticsData {
  macro: {
    columns: string[];
    rows: MacroRow[];
  };
  species: SpeciesItem[];
  architecture: ArchitectureSpec[];
}

export interface SampleImage {
  id: string;
  name: string;
  description: string;
  url: string;
}
