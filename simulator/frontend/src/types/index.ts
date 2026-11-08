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

export interface MacroFold0Row {
  metric: string;
  paper: number;
  recreation: number;
  enhanced: number;
  delta_vs_paper: number;
  delta_vs_recreation: number;
  unit: string;
  higher_is_better: boolean;
}

export interface MacroFiveFoldRow {
  metric: string;
  paper: string;
  recreation: string;
  enhanced: string;
  delta_vs_paper: string;
  delta_vs_recreation: string;
  higher_is_better: boolean;
}

export interface FoldDetail {
  fold: number;
  precision: number;
  recall: number;
  map50: number;
  map50_95: number;
  f1: number;
  is_active_checkpoint: boolean;
}

export interface SpeciesItem {
  class_id: number;
  species: string;
  order: string;
  paper_map50: number;
  recreation_map50: number;
  recreation_map50_std: number;
  recreation_p: number;
  recreation_r: number;
  recreation_map50_95: number;
  enhanced_map50: number;
  enhanced_map50_std: number;
  enhanced_p: number;
  enhanced_r: number;
  enhanced_map50_95: number;
  delta_vs_paper: number;
  delta_vs_recreation: number;
}

export interface ArchitectureSpec {
  component: string;
  baseline: string;
  improved: string;
}

export interface StatisticsData {
  macro: {
    columns: string[];
    fold0_rows: MacroFold0Row[];
    five_fold_rows: MacroFiveFoldRow[];
    folds: {
      enhanced: FoldDetail[];
      recreation: FoldDetail[];
    };
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
