import React, { useState, useMemo } from 'react';
import {
  BarChart2,
  Cpu,
  Download,
  Layers,
  Search,
  ArrowUpDown,
} from 'lucide-react';
import type {
  ArchitectureSpec,
  FoldDetail,
  MacroFiveFoldRow,
  MacroFold0Row,
  SpeciesItem,
  StatisticsData,
} from '../types';

interface StatisticsDashboardProps {
  stats: StatisticsData | null;
}

export const StatisticsDashboard: React.FC<StatisticsDashboardProps> = ({ stats }) => {
  const [macroMode, setMacroMode] = useState<'fold0' | 'fivefold' | 'folds'>('fold0');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedOrder, setSelectedOrder] = useState<string>('all');
  const [sortBy, setSortBy] = useState<
    'species' | 'enhanced_map50' | 'delta_vs_paper' | 'delta_vs_recreation'
  >('enhanced_map50');
  const [sortAsc, setSortAsc] = useState(false);

  // Available taxonomic orders
  const orders = useMemo(() => {
    if (!stats) return ['all'];
    const set = new Set(stats.species.map((s) => s.order));
    return ['all', ...Array.from(set)];
  }, [stats]);

  // Filtered and sorted species
  const filteredSpecies = useMemo(() => {
    if (!stats) return [];
    return stats.species
      .filter((s) => {
        const matchesSearch =
          s.species.toLowerCase().includes(searchTerm.toLowerCase()) ||
          s.order.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesOrder = selectedOrder === 'all' || s.order.toLowerCase() === selectedOrder.toLowerCase();
        return matchesSearch && matchesOrder;
      })
      .sort((a, b) => {
        const valA: any = a[sortBy];
        const valB: any = b[sortBy];
        if (typeof valA === 'string') {
          return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
        }
        return sortAsc ? valA - valB : valB - valA;
      });
  }, [stats, searchTerm, selectedOrder, sortBy, sortAsc]);

  const toggleSort = (
    field: 'species' | 'enhanced_map50' | 'delta_vs_paper' | 'delta_vs_recreation'
  ) => {
    if (sortBy === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortBy(field);
      setSortAsc(false);
    }
  };

  // Export to CSV
  const handleExportCSV = () => {
    if (!stats) return;
    const headers = [
      'Class ID',
      'Species',
      'Order',
      'Paper mAP50 (%)',
      'Recreation mAP50 (%)',
      'Enhanced mAP50 (%)',
      'Enhanced mAP50 Std',
      'Enhanced Precision (%)',
      'Enhanced Recall (%)',
      'Enhanced mAP50-95 (%)',
      'Delta vs Paper (%)',
      'Delta vs Recreation (%)',
    ];

    const rows = stats.species.map((s) => [
      s.class_id,
      `"${s.species}"`,
      `"${s.order}"`,
      s.paper_map50.toFixed(2),
      s.recreation_map50.toFixed(2),
      s.enhanced_map50.toFixed(2),
      s.enhanced_map50_std.toFixed(2),
      s.enhanced_p.toFixed(2),
      s.enhanced_r.toFixed(2),
      s.enhanced_map50_95.toFixed(2),
      s.delta_vs_paper.toFixed(2),
      s.delta_vs_recreation.toFixed(2),
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'PaddyYOLO_Species_Benchmark.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!stats) {
    return (
      <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-12 text-center">
        <p className="text-sm text-[#8b949e]">Loading benchmark statistics...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 1. Scientific Research KPI Banners */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
        <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-4">
          <div className="flex items-center justify-between text-xs text-[#8b949e] mb-1">
            <span className="font-medium uppercase tracking-wider text-[10px]">mAP@50 Gain</span>
            <span className="font-mono text-white text-[11px]">+11.28% vs Rec</span>
          </div>
          <div className="flex items-baseline space-x-1.5 font-mono">
            <span className="text-2xl font-bold text-white tabular-nums">78.83%</span>
            <span className="text-xs text-[#3fb950] font-medium">(+8.63% vs Paper)</span>
          </div>
          <div className="mt-2 text-[10px] text-[#656d76] font-mono">
            Fold 0 Active Checkpoint · Baseline: 67.55%
          </div>
        </div>

        <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-4">
          <div className="flex items-center justify-between text-xs text-[#8b949e] mb-1">
            <span className="font-medium uppercase tracking-wider text-[10px]">mAP@50-95 Strict</span>
            <span className="font-mono text-white text-[11px]">+13.41% vs Rec</span>
          </div>
          <div className="flex items-baseline space-x-1.5 font-mono">
            <span className="text-2xl font-bold text-white tabular-nums">61.16%</span>
            <span className="text-xs text-[#3fb950] font-medium">(+9.66% vs Paper)</span>
          </div>
          <div className="mt-2 text-[10px] text-[#656d76] font-mono">
            Higher IoU threshold bounding box alignment
          </div>
        </div>

        <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-4">
          <div className="flex items-center justify-between text-xs text-[#8b949e] mb-1">
            <span className="font-medium uppercase tracking-wider text-[10px]">Small Pest Recall</span>
            <span className="font-mono text-white text-[11px]">+9.68%</span>
          </div>
          <div className="flex items-baseline space-x-1.5 font-mono">
            <span className="text-2xl font-bold text-white tabular-nums">71.50%</span>
            <span className="text-xs text-[#3fb950] font-medium">(+8.50% vs Paper)</span>
          </div>
          <div className="mt-2 text-[10px] text-[#656d76] font-mono">
            Reduction in false negatives on rice tillers
          </div>
        </div>

        <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-4">
          <div className="flex items-center justify-between text-xs text-[#8b949e] mb-1">
            <span className="font-medium uppercase tracking-wider text-[10px]">5-Fold Cross-Val</span>
            <span className="text-[#8b949e] font-mono text-[11px]">μ ± σ</span>
          </div>
          <div className="flex items-baseline space-x-1.5 font-mono">
            <span className="text-2xl font-bold text-white tabular-nums">77.45%</span>
            <span className="text-xs text-[#8b949e]">± 3.25%</span>
          </div>
          <div className="mt-2 text-[10px] text-[#656d76] font-mono">
            Full 2,024 images 5-partition holdout evaluation
          </div>
        </div>
      </div>

      {/* 2. Macro Benchmark Evaluation (Paper vs Recreation vs Enhanced) */}
      <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-lg bg-[#171e28] border border-[#1c2432] flex items-center justify-center text-[#8b949e]">
              <BarChart2 className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="text-sm font-semibold text-white tracking-tight">
                  Macro Benchmark Evaluation
                </h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#171e28] border border-[#1c2432] text-[#8b949e]">
                  Table 8 Benchmark Audit
                </span>
              </div>
              <p className="text-xs text-[#656d76]">
                Published Paper (Zheng et al., 2025) vs Baseline Recreation vs Enhanced Paddy-YOLO
              </p>
            </div>
          </div>

          {/* Sub-view toggle */}
          <div className="flex items-center p-1 bg-[#171e28] rounded-lg border border-[#1c2432] text-xs">
            <button
              onClick={() => setMacroMode('fold0')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                macroMode === 'fold0'
                  ? 'bg-[#1f2735] text-white shadow-xs'
                  : 'text-[#8b949e] hover:text-white'
              }`}
            >
              Fold 0 (Active Checkpoint)
            </button>
            <button
              onClick={() => setMacroMode('fivefold')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                macroMode === 'fivefold'
                  ? 'bg-[#1f2735] text-white shadow-xs'
                  : 'text-[#8b949e] hover:text-white'
              }`}
            >
              5-Fold Cross-Val (μ ± σ)
            </button>
            <button
              onClick={() => setMacroMode('folds')}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                macroMode === 'folds'
                  ? 'bg-[#1f2735] text-white shadow-xs'
                  : 'text-[#8b949e] hover:text-white'
              }`}
            >
              Individual Folds
            </button>
          </div>
        </div>

        {/* View A: Fold 0 Active Checkpoint Evaluation */}
        {macroMode === 'fold0' && (
          <div className="space-y-3">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#1c2432] text-[#8b949e] uppercase tracking-wider font-semibold text-[11px]">
                    <th className="py-2.5 px-3">Metric</th>
                    <th className="py-2.5 px-3 text-right">Paper (Published)</th>
                    <th className="py-2.5 px-3 text-right">Recreation (Baseline)</th>
                    <th className="py-2.5 px-3 text-right text-white">Enhanced (Fold 0)</th>
                    <th className="py-2.5 px-3 text-right">Δ vs Paper</th>
                    <th className="py-2.5 px-3 text-right">Δ vs Recreation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1c2432]/60 font-mono tabular-nums">
                  {stats.macro.fold0_rows.map((row: MacroFold0Row, i: number) => {
                    const isPosPaper = row.delta_vs_paper > 0;
                    const isPosRec = row.delta_vs_recreation > 0;
                    return (
                      <tr key={i} className="hover:bg-[#171e28]/50 transition-colors">
                        <td className="py-2.5 px-3 font-sans font-medium text-white">
                          {row.metric}
                        </td>
                        <td className="py-2.5 px-3 text-right text-[#8b949e]">
                          {row.paper.toFixed(2)}{row.unit}
                        </td>
                        <td className="py-2.5 px-3 text-right text-[#8b949e]">
                          {row.recreation.toFixed(2)}{row.unit}
                        </td>
                        <td className="py-2.5 px-3 text-right font-medium text-white text-[13px]">
                          {row.enhanced.toFixed(2)}{row.unit}
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                              isPosPaper
                                ? 'bg-[#1a2e22] text-[#3fb950] border border-[#23422e]'
                                : 'bg-[#2d1b1f] text-[#f85149] border border-[#432328]'
                            }`}
                          >
                            {isPosPaper ? `+${row.delta_vs_paper.toFixed(2)}` : row.delta_vs_paper.toFixed(2)}{row.unit}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                              isPosRec
                                ? 'bg-[#1a2e22] text-[#3fb950] border border-[#23422e]'
                                : 'bg-[#2d1b1f] text-[#f85149] border border-[#432328]'
                            }`}
                          >
                            {isPosRec ? `+${row.delta_vs_recreation.toFixed(2)}` : row.delta_vs_recreation.toFixed(2)}{row.unit}
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
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#1c2432] text-[#8b949e] uppercase tracking-wider font-semibold text-[11px]">
                    <th className="py-2.5 px-3">Metric</th>
                    <th className="py-2.5 px-3 text-right">Paper (Published)</th>
                    <th className="py-2.5 px-3 text-right">Recreation (5-Fold μ ± σ)</th>
                    <th className="py-2.5 px-3 text-right text-white">Enhanced (5-Fold μ ± σ)</th>
                    <th className="py-2.5 px-3 text-right">Δ vs Paper</th>
                    <th className="py-2.5 px-3 text-right">Δ vs Recreation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1c2432]/60 font-mono tabular-nums">
                  {stats.macro.five_fold_rows.map((row: MacroFiveFoldRow, i: number) => (
                    <tr key={i} className="hover:bg-[#171e28]/50 transition-colors">
                      <td className="py-2.5 px-3 font-sans font-medium text-white">
                        {row.metric}
                      </td>
                      <td className="py-2.5 px-3 text-right text-[#8b949e]">
                        {row.paper}
                      </td>
                      <td className="py-2.5 px-3 text-right text-[#8b949e]">
                        {row.recreation}
                      </td>
                      <td className="py-2.5 px-3 text-right font-medium text-white text-[13px]">
                        {row.enhanced}
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-[#1a2e22] text-[#3fb950] border border-[#23422e]">
                          {row.delta_vs_paper}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-[#1a2e22] text-[#3fb950] border border-[#23422e]">
                          {row.delta_vs_recreation}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* View C: Individual Folds Detailed Breakdown */}
        {macroMode === 'folds' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 pt-1">
            {/* Recreation Folds */}
            <div className="bg-[#0b0f14] rounded-lg border border-[#1c2432] p-3.5">
              <div className="flex items-center justify-between pb-2 border-b border-[#1c2432] mb-2.5">
                <span className="text-xs font-medium text-white uppercase tracking-wider">
                  Baseline Recreation (5 Folds)
                </span>
                <span className="text-[10px] font-mono text-[#656d76]">Paddy-YOLO</span>
              </div>
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#1c2432] text-[#8b949e] font-mono text-[10px]">
                    <th className="py-2 px-2">Fold</th>
                    <th className="py-2 px-2 text-right">P (%)</th>
                    <th className="py-2 px-2 text-right">R (%)</th>
                    <th className="py-2 px-2 text-right">mAP@50</th>
                    <th className="py-2 px-2 text-right">mAP@50-95</th>
                    <th className="py-2 px-2 text-right">F1</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1c2432]/40 font-mono tabular-nums text-[11px]">
                  {stats.macro.folds.recreation.map((f: FoldDetail) => (
                    <tr
                      key={f.fold}
                      className={f.is_active_checkpoint ? 'bg-[#171e28] font-medium' : 'hover:bg-[#171e28]/40'}
                    >
                      <td className="py-2 px-2 text-white flex items-center space-x-1">
                        <span>Fold {f.fold}</span>
                        {f.is_active_checkpoint && (
                          <span className="text-[9px] px-1 py-0.2 rounded bg-[#1c2432] text-[#8b949e] border border-[#283344]">
                            best.pt
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-2 text-right text-[#8b949e]">{f.precision.toFixed(1)}</td>
                      <td className="py-2 px-2 text-right text-[#8b949e]">{f.recall.toFixed(1)}</td>
                      <td className="py-2 px-2 text-right text-white font-medium">{f.map50.toFixed(2)}%</td>
                      <td className="py-2 px-2 text-right text-[#8b949e]">{f.map50_95.toFixed(2)}%</td>
                      <td className="py-2 px-2 text-right text-[#8b949e]">{f.f1.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Enhanced Folds */}
            <div className="bg-[#0b0f14] rounded-lg border border-[#1c2432] p-3.5">
              <div className="flex items-center justify-between pb-2 border-b border-[#1c2432] mb-2.5">
                <span className="text-xs font-medium text-white uppercase tracking-wider">
                  Enhanced Architecture (5 Folds)
                </span>
                <span className="text-[10px] font-mono text-[#656d76]">EMA + DySample</span>
              </div>
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[#1c2432] text-[#8b949e] font-mono text-[10px]">
                    <th className="py-2 px-2">Fold</th>
                    <th className="py-2 px-2 text-right">P (%)</th>
                    <th className="py-2 px-2 text-right">R (%)</th>
                    <th className="py-2 px-2 text-right">mAP@50</th>
                    <th className="py-2 px-2 text-right">mAP@50-95</th>
                    <th className="py-2 px-2 text-right">F1</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1c2432]/40 font-mono tabular-nums text-[11px]">
                  {stats.macro.folds.enhanced.map((f: FoldDetail) => (
                    <tr
                      key={f.fold}
                      className={f.is_active_checkpoint ? 'bg-[#171e28] font-medium' : 'hover:bg-[#171e28]/40'}
                    >
                      <td className="py-2 px-2 text-white flex items-center space-x-1">
                        <span>Fold {f.fold}</span>
                        {f.is_active_checkpoint && (
                          <span className="text-[9px] px-1 py-0.2 rounded bg-[#1c2432] text-white border border-[#283344]">
                            best.pt
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-2 text-right text-[#8b949e]">{f.precision.toFixed(1)}</td>
                      <td className="py-2 px-2 text-right text-[#8b949e]">{f.recall.toFixed(1)}</td>
                      <td className="py-2 px-2 text-right text-white font-medium">{f.map50.toFixed(2)}%</td>
                      <td className="py-2 px-2 text-right text-[#8b949e]">{f.map50_95.toFixed(2)}%</td>
                      <td className="py-2 px-2 text-right text-[#8b949e]">{f.f1.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* 3. 16 Rice Pest Species Detailed Benchmark */}
      <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-5 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-4">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-lg bg-[#171e28] border border-[#1c2432] flex items-center justify-center text-[#8b949e]">
              <Layers className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white tracking-tight">
                16 Rice Pest Species Detailed Benchmark
              </h3>
              <p className="text-xs text-[#656d76]">
                Species-level detection accuracy (mAP@50 μ ± σ) comparing Paper, Recreation, and Enhanced Paddy-YOLO
              </p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-[#656d76]" />
              <input
                type="text"
                placeholder="Search species..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="bg-[#0b0f14] text-xs text-white pl-8 pr-3 py-1.5 rounded-lg border border-[#1c2432] focus:outline-none focus:border-[#388bfd] w-40 placeholder-[#656d76]"
              />
            </div>

            <div className="flex items-center p-0.5 bg-[#0b0f14] rounded-lg border border-[#1c2432] text-[11px]">
              {orders.map((ord) => (
                <button
                  key={ord}
                  onClick={() => setSelectedOrder(ord)}
                  className={`px-2 py-1 rounded capitalize transition-colors ${
                    selectedOrder.toLowerCase() === ord.toLowerCase()
                      ? 'bg-[#1f2735] text-white font-medium'
                      : 'text-[#656d76] hover:text-[#8b949e]'
                  }`}
                >
                  {ord}
                </button>
              ))}
            </div>

            <button
              onClick={handleExportCSV}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-[#171e28] hover:bg-[#1f2735] border border-[#1c2432] hover:border-[#283344] text-xs font-medium text-white transition-colors"
              title="Export complete 16 species dataset to CSV"
            >
              <Download className="h-3.5 w-3.5 text-[#8b949e]" />
              <span>Export CSV</span>
            </button>
          </div>
        </div>

        {/* Species Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#1c2432] text-[#8b949e] uppercase tracking-wider font-semibold text-[11px]">
                <th
                  onClick={() => toggleSort('species')}
                  className="py-2.5 px-3 cursor-pointer hover:text-white transition-colors"
                >
                  <div className="flex items-center space-x-1">
                    <span>Species</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th className="py-2.5 px-2.5">Order</th>
                <th className="py-2.5 px-2.5 text-right">Paper mAP@50</th>
                <th className="py-2.5 px-2.5 text-right">Rec mAP@50</th>
                <th
                  onClick={() => toggleSort('enhanced_map50')}
                  className="py-2.5 px-3 text-right text-white cursor-pointer hover:text-[#8b949e] transition-colors"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>Enhanced mAP@50</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th className="py-2.5 px-2.5 text-right">P / R</th>
                <th className="py-2.5 px-2.5 text-right">mAP@50-95</th>
                <th
                  onClick={() => toggleSort('delta_vs_paper')}
                  className="py-2.5 px-3 text-right cursor-pointer hover:text-white transition-colors"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>Δ vs Paper</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th
                  onClick={() => toggleSort('delta_vs_recreation')}
                  className="py-2.5 px-3 text-right cursor-pointer hover:text-white transition-colors"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>Δ vs Rec</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th className="py-2.5 px-3">Visual Gain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1c2432]/60 font-mono tabular-nums">
              {filteredSpecies.map((item: SpeciesItem) => {
                const isPosPaper = item.delta_vs_paper > 0;
                const isPosRec = item.delta_vs_recreation > 0;
                const gainPct = Math.min(100, Math.max(0, (item.delta_vs_recreation / 20) * 100));

                return (
                  <tr key={item.class_id} className="hover:bg-[#171e28]/50 transition-colors">
                    <td className="py-2.5 px-3 font-sans font-medium text-white capitalize">
                      {item.species}
                    </td>
                    <td className="py-2.5 px-2.5 text-[#656d76] font-sans text-[11px]">
                      {item.order}
                    </td>
                    <td className="py-2.5 px-2.5 text-right text-[#8b949e]">
                      {item.paper_map50.toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-2.5 text-right text-[#8b949e]">
                      {item.recreation_map50.toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-right font-medium text-white text-[13px]">
                      {item.enhanced_map50.toFixed(1)}%
                      <span className="text-[10px] text-[#656d76] ml-1 font-normal">
                        ±{item.enhanced_map50_std.toFixed(1)}
                      </span>
                    </td>
                    <td className="py-2.5 px-2.5 text-right text-[#8b949e] text-[11px]">
                      {item.enhanced_p.toFixed(0)}% / {item.enhanced_r.toFixed(0)}%
                    </td>
                    <td className="py-2.5 px-2.5 text-right text-[#8b949e]">
                      {item.enhanced_map50_95.toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span
                        className={`inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-medium ${
                          isPosPaper
                            ? 'bg-[#1a2e22] text-[#3fb950] border border-[#23422e]'
                            : 'bg-[#2d1b1f] text-[#f85149] border border-[#432328]'
                        }`}
                      >
                        {isPosPaper ? `+${item.delta_vs_paper.toFixed(1)}%` : `${item.delta_vs_paper.toFixed(1)}%`}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span
                        className={`inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-medium ${
                          isPosRec
                            ? 'bg-[#1a2e22] text-[#3fb950] border border-[#23422e]'
                            : 'bg-[#2d1b1f] text-[#f85149] border border-[#432328]'
                        }`}
                      >
                        {isPosRec ? `+${item.delta_vs_recreation.toFixed(1)}%` : `${item.delta_vs_recreation.toFixed(1)}%`}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 min-w-[80px]">
                      <div className="h-1.5 w-full bg-[#1c2432] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#3fb950] rounded-full opacity-80"
                          style={{ width: `${gainPct}%` }}
                        />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. Architectural Innovations & Ablation Matrix */}
      <div className="bg-[#11161d] rounded-xl border border-[#1c2432] p-5 shadow-xs">
        <div className="flex items-center space-x-3 mb-4">
          <div className="h-8 w-8 rounded-lg bg-[#171e28] border border-[#1c2432] flex items-center justify-center text-[#8b949e]">
            <Cpu className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white tracking-tight">
              Architectural Specifications & Ablation Rationale
            </h3>
            <p className="text-xs text-[#656d76]">
              Structural comparison of baseline Paddy-YOLO vs EMA attention and DySample dynamic upsampling
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3.5">
          {stats.architecture.map((item: ArchitectureSpec, i: number) => (
            <div
              key={i}
              className="bg-[#0b0f14] rounded-lg border border-[#1c2432] p-3.5 flex flex-col justify-between"
            >
              <span className="text-[10px] font-semibold text-[#8b949e] uppercase tracking-wider mb-2 block">
                {item.component}
              </span>
              <div className="space-y-2">
                <div>
                  <span className="text-[10px] text-[#8b949e] block font-mono">Baseline (Paddy YOLO):</span>
                  <span className="text-xs font-mono text-[#8b949e]">{item.baseline}</span>
                </div>
                <div className="pt-2 border-t border-[#1c2432]">
                  <span className="text-[10px] text-white block font-mono">Enhanced (EMA + DySample):</span>
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
