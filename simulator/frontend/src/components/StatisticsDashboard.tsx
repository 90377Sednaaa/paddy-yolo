import React, { useState } from 'react';
import { BarChart2, CheckCircle2, Cpu, Filter, Layers, Search } from 'lucide-react';
import type { ArchitectureSpec, FoldDetail, MacroFiveFoldRow, MacroFold0Row, SpeciesItem, StatisticsData } from '../types';

interface StatisticsDashboardProps {
  stats: StatisticsData | null;
}

export const StatisticsDashboard: React.FC<StatisticsDashboardProps> = ({ stats }) => {
  const [macroMode, setMacroMode] = useState<'fold0' | 'fivefold' | 'folds'>('fold0');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'species' | 'enhanced_map50' | 'delta_vs_paper' | 'delta_vs_recreation'>('enhanced_map50');
  const [sortAsc, setSortAsc] = useState(false);

  if (!stats) {
    return (
      <div className="bg-[#111622] rounded-xl border border-[#232d42] p-12 text-center">
        <p className="text-sm text-[#94a3b8]">Loading benchmark statistics...</p>
      </div>
    );
  }

  const filteredSpecies = stats.species
    .filter(
      (s) =>
        s.species.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.order.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .sort((a, b) => {
      const valA: any = a[sortBy];
      const valB: any = b[sortBy];
      if (typeof valA === 'string') {
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return sortAsc ? valA - valB : valB - valA;
    });

  const toggleSort = (field: 'species' | 'enhanced_map50' | 'delta_vs_paper' | 'delta_vs_recreation') => {
    if (sortBy === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortBy(field);
      setSortAsc(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* 1. Macro Benchmark Evaluation (Paper vs Recreation vs Enhanced) */}
      <div className="bg-[#111622] rounded-xl border border-[#232d42] p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-lg bg-[#182030] border border-[#232d42] flex items-center justify-center text-[#10b981]">
              <BarChart2 className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="text-base font-bold text-white tracking-tight">
                  Macro Benchmark Evaluation
                </h3>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#182030] border border-[#334155] text-[#10b981]">
                  Paper vs Recreation vs Enhanced
                </span>
              </div>
              <p className="text-xs text-[#64748b]">
                Comparative research evaluation across published Paper, baseline Recreation, and Enhanced Paddy-YOLO
              </p>
            </div>
          </div>

          {/* Sub-view toggle */}
          <div className="flex items-center p-1 bg-[#0a0d14] rounded-lg border border-[#232d42] text-xs">
            <button
              onClick={() => setMacroMode('fold0')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                macroMode === 'fold0'
                  ? 'bg-[#182030] text-white border border-[#334155] shadow-sm'
                  : 'text-[#94a3b8] hover:text-white'
              }`}
            >
              Fold 0 (Active Checkpoint)
            </button>
            <button
              onClick={() => setMacroMode('fivefold')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                macroMode === 'fivefold'
                  ? 'bg-[#182030] text-white border border-[#334155] shadow-sm'
                  : 'text-[#94a3b8] hover:text-white'
              }`}
            >
              5-Fold Macro Average (μ ± σ)
            </button>
            <button
              onClick={() => setMacroMode('folds')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                macroMode === 'folds'
                  ? 'bg-[#182030] text-white border border-[#334155] shadow-sm'
                  : 'text-[#94a3b8] hover:text-white'
              }`}
            >
              Individual Folds
            </button>
          </div>
        </div>

        {/* View A: Fold 0 Active Checkpoint Evaluation */}
        {macroMode === 'fold0' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-[#94a3b8] px-1">
              <span className="flex items-center space-x-1.5">
                <CheckCircle2 className="h-3.5 w-3.5 text-[#10b981]" />
                <span className="text-white font-medium">Active Weights:</span>
                <span>Fold 0 weights deployed as <code className="text-[#38bdf8] font-mono">best.pt</code> in the Vision Arena</span>
              </span>
              <span className="text-[11px] text-[#64748b] font-mono">Table 8 Baseline vs Enhanced</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#232d42] text-[#64748b] uppercase tracking-wider font-semibold">
                    <th className="py-3 px-4">Metric</th>
                    <th className="py-3 px-4 text-right">Paper (Published)</th>
                    <th className="py-3 px-4 text-right">Recreation (Fold 0)</th>
                    <th className="py-3 px-4 text-right text-[#10b981]">Enhanced (Fold 0)</th>
                    <th className="py-3 px-4 text-right">Δ vs Paper</th>
                    <th className="py-3 px-4 text-right">Δ vs Recreation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#232d42]/60 font-mono tabular-nums">
                  {stats.macro.fold0_rows.map((row: MacroFold0Row, i: number) => {
                    const isPosPaper = row.delta_vs_paper > 0;
                    const isPosRec = row.delta_vs_recreation > 0;
                    return (
                      <tr key={i} className="hover:bg-[#182030]/40 transition-colors">
                        <td className="py-3.5 px-4 font-sans font-medium text-white">
                          {row.metric}
                        </td>
                        <td className="py-3.5 px-4 text-right text-[#94a3b8]">
                          {row.paper.toFixed(2)}{row.unit}
                        </td>
                        <td className="py-3.5 px-4 text-right text-[#94a3b8]">
                          {row.recreation.toFixed(2)}{row.unit}
                        </td>
                        <td className="py-3.5 px-4 text-right font-bold text-white text-[13px]">
                          {row.enhanced.toFixed(2)}{row.unit}
                        </td>
                        <td className="py-3.5 px-4 text-right">
                          <span
                            className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-xs font-semibold ${
                              isPosPaper
                                ? 'bg-[#10b981]/10 text-[#10b981] border border-[#10b981]/20'
                                : 'bg-[#ef4444]/10 text-[#ef4444] border border-[#ef4444]/20'
                            }`}
                          >
                            <span>{isPosPaper ? `+${row.delta_vs_paper.toFixed(2)}` : row.delta_vs_paper.toFixed(2)}{row.unit}</span>
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-right">
                          <span
                            className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-xs font-semibold ${
                              isPosRec
                                ? 'bg-[#10b981]/10 text-[#10b981] border border-[#10b981]/20'
                                : 'bg-[#ef4444]/10 text-[#ef4444] border border-[#ef4444]/20'
                            }`}
                          >
                            <span>{isPosRec ? `+${row.delta_vs_recreation.toFixed(2)}` : row.delta_vs_recreation.toFixed(2)}{row.unit}</span>
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* View B: 5-Fold Cross-Validation Macro Average (μ ± σ) */}
        {macroMode === 'fivefold' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-[#94a3b8] px-1">
              <span>Holdout test partition evaluations averaged across all 5 cross-validation splits</span>
              <span className="text-[11px] text-[#64748b] font-mono">Dataset R2000 (2024 Images)</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#232d42] text-[#64748b] uppercase tracking-wider font-semibold">
                    <th className="py-3 px-4">Metric</th>
                    <th className="py-3 px-4 text-right">Paper (Published)</th>
                    <th className="py-3 px-4 text-right">Recreation (5-Fold μ ± σ)</th>
                    <th className="py-3 px-4 text-right text-[#10b981]">Enhanced (5-Fold μ ± σ)</th>
                    <th className="py-3 px-4 text-right">Δ vs Paper</th>
                    <th className="py-3 px-4 text-right">Δ vs Recreation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#232d42]/60 font-mono tabular-nums">
                  {stats.macro.five_fold_rows.map((row: MacroFiveFoldRow, i: number) => {
                    return (
                      <tr key={i} className="hover:bg-[#182030]/40 transition-colors">
                        <td className="py-3.5 px-4 font-sans font-medium text-white">
                          {row.metric}
                        </td>
                        <td className="py-3.5 px-4 text-right text-[#94a3b8]">
                          {row.paper}
                        </td>
                        <td className="py-3.5 px-4 text-right text-[#94a3b8]">
                          {row.recreation}
                        </td>
                        <td className="py-3.5 px-4 text-right font-bold text-white text-[13px]">
                          {row.enhanced}
                        </td>
                        <td className="py-3.5 px-4 text-right">
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-[#10b981]/10 text-[#10b981] border border-[#10b981]/20">
                            {row.delta_vs_paper}
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-right">
                          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-[#10b981]/10 text-[#10b981] border border-[#10b981]/20">
                            {row.delta_vs_recreation}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* View C: Individual Folds Detailed Breakdown */}
        {macroMode === 'folds' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2">
            {/* Recreation Folds */}
            <div className="bg-[#0a0d14] rounded-lg border border-[#232d42] p-4">
              <div className="flex items-center justify-between pb-2 border-b border-[#232d42] mb-3">
                <span className="text-xs font-bold text-white uppercase tracking-wider">
                  Recreation (Baseline Paddy-YOLO)
                </span>
                <span className="text-[11px] font-mono text-[#64748b]">5 Folds</span>
              </div>
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#232d42] text-[#64748b] font-mono text-[11px]">
                    <th className="py-2 px-2">Fold</th>
                    <th className="py-2 px-2 text-right">P (%)</th>
                    <th className="py-2 px-2 text-right">R (%)</th>
                    <th className="py-2 px-2 text-right">mAP@50</th>
                    <th className="py-2 px-2 text-right">mAP@50-95</th>
                    <th className="py-2 px-2 text-right">F1</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#232d42]/40 font-mono tabular-nums text-[11px]">
                  {stats.macro.folds.recreation.map((f: FoldDetail) => (
                    <tr
                      key={f.fold}
                      className={f.is_active_checkpoint ? 'bg-[#182030]/60 font-semibold' : 'hover:bg-[#182030]/20'}
                    >
                      <td className="py-2 px-2 text-white flex items-center space-x-1">
                        <span>Fold {f.fold}</span>
                        {f.is_active_checkpoint && (
                          <span className="text-[9px] px-1 py-0.2 rounded bg-[#38bdf8]/10 text-[#38bdf8] border border-[#38bdf8]/20">
                            best.pt
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-2 text-right text-[#94a3b8]">{f.precision.toFixed(1)}</td>
                      <td className="py-2 px-2 text-right text-[#94a3b8]">{f.recall.toFixed(1)}</td>
                      <td className="py-2 px-2 text-right text-white font-medium">{f.map50.toFixed(2)}%</td>
                      <td className="py-2 px-2 text-right text-[#94a3b8]">{f.map50_95.toFixed(2)}%</td>
                      <td className="py-2 px-2 text-right text-[#94a3b8]">{f.f1.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Enhanced Folds */}
            <div className="bg-[#0a0d14] rounded-lg border border-[#232d42] p-4">
              <div className="flex items-center justify-between pb-2 border-b border-[#232d42] mb-3">
                <span className="text-xs font-bold text-[#10b981] uppercase tracking-wider">
                  Enhanced (Paddy-YOLO EMA + DySample)
                </span>
                <span className="text-[11px] font-mono text-[#64748b]">5 Folds</span>
              </div>
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#232d42] text-[#64748b] font-mono text-[11px]">
                    <th className="py-2 px-2">Fold</th>
                    <th className="py-2 px-2 text-right">P (%)</th>
                    <th className="py-2 px-2 text-right">R (%)</th>
                    <th className="py-2 px-2 text-right">mAP@50</th>
                    <th className="py-2 px-2 text-right">mAP@50-95</th>
                    <th className="py-2 px-2 text-right">F1</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#232d42]/40 font-mono tabular-nums text-[11px]">
                  {stats.macro.folds.enhanced.map((f: FoldDetail) => (
                    <tr
                      key={f.fold}
                      className={f.is_active_checkpoint ? 'bg-[#182030]/60 font-semibold' : 'hover:bg-[#182030]/20'}
                    >
                      <td className="py-2 px-2 text-white flex items-center space-x-1">
                        <span>Fold {f.fold}</span>
                        {f.is_active_checkpoint && (
                          <span className="text-[9px] px-1 py-0.2 rounded bg-[#10b981]/10 text-[#10b981] border border-[#10b981]/20">
                            best.pt
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-2 text-right text-[#94a3b8]">{f.precision.toFixed(1)}</td>
                      <td className="py-2 px-2 text-right text-[#94a3b8]">{f.recall.toFixed(1)}</td>
                      <td className="py-2 px-2 text-right text-[#10b981] font-bold">{f.map50.toFixed(2)}%</td>
                      <td className="py-2 px-2 text-right text-white font-medium">{f.map50_95.toFixed(2)}%</td>
                      <td className="py-2 px-2 text-right text-[#94a3b8]">{f.f1.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* 2. 16 Rice Pest Species Detailed Benchmark */}
      <div className="bg-[#111622] rounded-xl border border-[#232d42] p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-lg bg-[#182030] border border-[#232d42] flex items-center justify-center text-[#10b981]">
              <Layers className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white tracking-tight">
                16 Rice Pest Species Detailed Benchmark
              </h3>
              <p className="text-xs text-[#64748b]">
                Species-level detection accuracy (mAP@50 μ ± σ) comparing Paper, Recreation, and Enhanced Paddy-YOLO
              </p>
            </div>
          </div>

          {/* Search bar */}
          <div className="relative">
            <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-[#64748b]" />
            <input
              type="text"
              placeholder="Filter pest species..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-[#182030] text-xs text-white pl-9 pr-4 py-2 rounded-lg border border-[#232d42] focus:outline-none focus:border-[#10b981] w-56 placeholder-[#64748b]"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#232d42] text-[#64748b] uppercase tracking-wider font-semibold">
                <th
                  onClick={() => toggleSort('species')}
                  className="py-3 px-4 cursor-pointer hover:text-white transition-colors"
                >
                  <div className="flex items-center space-x-1">
                    <span>Species</span>
                    <Filter className="h-3 w-3" />
                  </div>
                </th>
                <th className="py-3 px-3">Order</th>
                <th className="py-3 px-3 text-right">Paper mAP@50</th>
                <th className="py-3 px-3 text-right">Recreation mAP@50</th>
                <th
                  onClick={() => toggleSort('enhanced_map50')}
                  className="py-3 px-3 text-right text-[#10b981] cursor-pointer hover:text-white transition-colors"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>Enhanced mAP@50</span>
                    <Filter className="h-3 w-3" />
                  </div>
                </th>
                <th className="py-3 px-3 text-right">Enhanced P / R</th>
                <th className="py-3 px-3 text-right">mAP@50-95</th>
                <th
                  onClick={() => toggleSort('delta_vs_paper')}
                  className="py-3 px-3 text-right cursor-pointer hover:text-white transition-colors"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>Δ vs Paper</span>
                    <Filter className="h-3 w-3" />
                  </div>
                </th>
                <th
                  onClick={() => toggleSort('delta_vs_recreation')}
                  className="py-3 px-3 text-right cursor-pointer hover:text-white transition-colors"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>Δ vs Rec</span>
                    <Filter className="h-3 w-3" />
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#232d42]/60 font-mono tabular-nums">
              {filteredSpecies.map((item: SpeciesItem) => {
                const isPosPaper = item.delta_vs_paper > 0;
                const isPosRec = item.delta_vs_recreation > 0;
                return (
                  <tr key={item.class_id} className="hover:bg-[#182030]/40 transition-colors">
                    <td className="py-3 px-4 font-sans font-medium text-white capitalize">
                      {item.species}
                    </td>
                    <td className="py-3 px-3 text-[#64748b] font-sans text-[11px]">
                      {item.order}
                    </td>
                    <td className="py-3 px-3 text-right text-[#94a3b8]">
                      {item.paper_map50.toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-right text-[#94a3b8]">
                      {item.recreation_map50.toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-right font-bold text-white text-[13px]">
                      {item.enhanced_map50.toFixed(1)}%
                      <span className="text-[10px] text-[#64748b] ml-1 font-normal">
                        ±{item.enhanced_map50_std.toFixed(1)}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right text-[#94a3b8] text-[11px]">
                      {item.enhanced_p.toFixed(0)}% / {item.enhanced_r.toFixed(0)}%
                    </td>
                    <td className="py-3 px-3 text-right text-[#94a3b8]">
                      {item.enhanced_map50_95.toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-right">
                      <span
                        className={`inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-semibold ${
                          isPosPaper
                            ? 'bg-[#10b981]/10 text-[#10b981]'
                            : 'bg-[#ef4444]/10 text-[#ef4444]'
                        }`}
                      >
                        {isPosPaper ? `+${item.delta_vs_paper.toFixed(1)}%` : `${item.delta_vs_paper.toFixed(1)}%`}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right">
                      <span
                        className={`inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-semibold ${
                          isPosRec
                            ? 'bg-[#10b981]/10 text-[#10b981]'
                            : 'bg-[#ef4444]/10 text-[#ef4444]'
                        }`}
                      >
                        {isPosRec ? `+${item.delta_vs_recreation.toFixed(1)}%` : `${item.delta_vs_recreation.toFixed(1)}%`}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Architectural Specifications */}
      <div className="bg-[#111622] rounded-xl border border-[#232d42] p-6">
        <div className="flex items-center space-x-3 mb-6">
          <div className="h-8 w-8 rounded-lg bg-[#182030] border border-[#232d42] flex items-center justify-center text-[#10b981]">
            <Cpu className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">
              Architectural Specifications
            </h3>
            <p className="text-xs text-[#64748b]">
              Structural comparison of baseline Paddy-YOLO vs EMA attention and DySample dynamic upsampling
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {stats.architecture.map((item: ArchitectureSpec, i: number) => (
            <div key={i} className="bg-[#0a0d14] rounded-lg border border-[#232d42] p-3.5 flex flex-col justify-between">
              <span className="text-[11px] font-semibold text-[#64748b] uppercase tracking-wider mb-2 block">
                {item.component}
              </span>
              <div className="space-y-1.5">
                <div>
                  <span className="text-[10px] text-[#64748b] block font-mono">Baseline (Paddy YOLO):</span>
                  <span className="text-xs font-mono text-[#94a3b8]">{item.baseline}</span>
                </div>
                <div className="pt-1 border-t border-[#232d42]/60">
                  <span className="text-[10px] text-[#10b981] block font-mono">Enhanced (EMA + DySample):</span>
                  <span className="text-xs font-mono text-white font-medium">{item.improved}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
