import React, { useState } from 'react';
import { ArrowDown, ArrowUp, BarChart2, Cpu, Layers, Search } from 'lucide-react';
import type { ArchitectureSpec, MacroRow, SpeciesItem, StatisticsData } from '../types';

interface StatisticsDashboardProps {
  stats: StatisticsData | null;
}

export const StatisticsDashboard: React.FC<StatisticsDashboardProps> = ({ stats }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'species' | 'improved_map50' | 'delta_map50'>('improved_map50');
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

  const toggleSort = (field: 'species' | 'improved_map50' | 'delta_map50') => {
    if (sortBy === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortBy(field);
      setSortAsc(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* 1. Macro Evaluation Comparison */}
      <div className="bg-[#111622] rounded-xl border border-[#232d42] p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="h-8 w-8 rounded-lg bg-[#182030] border border-[#232d42] flex items-center justify-center text-[#10b981]">
            <BarChart2 className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">
              Macro Benchmark Evaluation
            </h3>
            <p className="text-xs text-[#64748b]">
              Comprehensive cross-validation metrics across all 16 rice pest categories
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#232d42] text-[#64748b] uppercase tracking-wider font-semibold">
                <th className="py-3 px-4">Metric</th>
                <th className="py-3 px-4 text-right">Paddy YOLO</th>
                <th className="py-3 px-4 text-right text-[#10b981]">
                  Paddy YOLO EMA + Dysample
                </th>
                <th className="py-3 px-4 text-right">Delta Improvement</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#232d42]/60 font-mono tabular-nums">
              {stats.macro.rows.map((row: MacroRow, i: number) => {
                const isPositive = row.delta > 0;
                return (
                  <tr key={i} className="hover:bg-[#182030]/40 transition-colors">
                    <td className="py-3.5 px-4 font-sans font-medium text-white">
                      {row.metric}
                    </td>
                    <td className="py-3.5 px-4 text-right text-[#94a3b8]">
                      {row.baseline.toFixed(2)}
                      {row.unit}
                    </td>
                    <td className="py-3.5 px-4 text-right font-bold text-white">
                      {row.improved.toFixed(2)}
                      {row.unit}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <span
                        className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-xs font-semibold ${
                          isPositive
                            ? 'bg-[#10b981]/10 text-[#10b981] border border-[#10b981]/20'
                            : row.delta === 0
                            ? 'bg-[#182030] text-[#94a3b8]'
                            : 'bg-[#ef4444]/10 text-[#ef4444] border border-[#ef4444]/20'
                        }`}
                      >
                        {isPositive ? (
                          <ArrowUp className="h-3 w-3 mr-0.5" />
                        ) : row.delta < 0 ? (
                          <ArrowDown className="h-3 w-3 mr-0.5" />
                        ) : null}
                        {row.delta > 0 ? `+${row.delta.toFixed(2)}` : row.delta.toFixed(2)}
                        {row.unit}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. 16 Rice Pest Species Detailed Breakdown */}
      <div className="bg-[#111622] rounded-xl border border-[#232d42] p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-lg bg-[#182030] border border-[#232d42] flex items-center justify-center text-[#38bdf8]">
              <Layers className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white tracking-tight">
                Species Detection Benchmark (16 Classes)
              </h3>
              <p className="text-xs text-[#64748b]">
                Species-level mAP@50 performance and comparative gains
              </p>
            </div>
          </div>

          {/* Search Filter */}
          <div className="relative w-full md:w-64">
            <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-[#64748b]" />
            <input
              type="text"
              placeholder="Filter by species or order..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-[#0a0d14] border border-[#232d42] rounded-lg pl-9 pr-3 py-1.5 text-xs text-white placeholder-[#64748b] focus:outline-none focus:border-[#334155]"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#232d42] text-[#64748b] uppercase tracking-wider font-semibold">
                <th
                  onClick={() => toggleSort('species')}
                  className="py-3 px-4 cursor-pointer hover:text-white"
                >
                  Species & Taxonomic Order
                </th>
                <th className="py-3 px-4 text-right">Paddy YOLO mAP@50</th>
                <th
                  onClick={() => toggleSort('improved_map50')}
                  className="py-3 px-4 text-right cursor-pointer hover:text-white text-[#10b981]"
                >
                  Paddy YOLO EMA + Dysample mAP@50
                </th>
                <th
                  onClick={() => toggleSort('delta_map50')}
                  className="py-3 px-4 text-right cursor-pointer hover:text-white"
                >
                  Delta Gain
                </th>
                <th className="py-3 px-4 text-right">Precision (%)</th>
                <th className="py-3 px-4 text-right">Recall (%)</th>
                <th className="py-3 px-4 text-right">mAP@50-95 (%)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#232d42]/60 font-mono tabular-nums">
              {filteredSpecies.map((s: SpeciesItem, i: number) => {
                const isPositive = s.delta_map50 > 0;
                return (
                  <tr key={i} className="hover:bg-[#182030]/40 transition-colors">
                    <td className="py-3 px-4 font-sans">
                      <div className="font-medium text-white capitalize">
                        {s.species}
                      </div>
                      <div className="text-[11px] text-[#64748b] italic">
                        {s.order}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-right text-[#94a3b8]">
                      {s.baseline_map50.toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right font-bold text-white">
                      {s.improved_map50.toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right">
                      <span
                        className={`inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-semibold ${
                          isPositive
                            ? 'bg-[#10b981]/10 text-[#10b981]'
                            : 'bg-[#ef4444]/10 text-[#ef4444]'
                        }`}
                      >
                        {isPositive ? `+${s.delta_map50.toFixed(2)}%` : `${s.delta_map50.toFixed(2)}%`}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right text-[#94a3b8]">
                      {s.improved_p.toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right text-[#94a3b8]">
                      {s.improved_r.toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 text-right text-[#94a3b8]">
                      {s.improved_map50_95.toFixed(2)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Architectural Specifications Comparison */}
      <div className="bg-[#111622] rounded-xl border border-[#232d42] p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="h-8 w-8 rounded-lg bg-[#182030] border border-[#232d42] flex items-center justify-center text-[#a855f7]">
            <Cpu className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">
              Architectural Specifications
            </h3>
            <p className="text-xs text-[#64748b]">
              Structural modifications and computational complexity comparison
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#232d42] text-[#64748b] uppercase tracking-wider font-semibold">
                <th className="py-3 px-4">Component / Dimension</th>
                <th className="py-3 px-4">Paddy YOLO</th>
                <th className="py-3 px-4 text-[#10b981]">
                  Paddy YOLO EMA + Dysample
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#232d42]/60 font-sans">
              {stats.architecture.map((item: ArchitectureSpec, i: number) => (
                <tr key={i} className="hover:bg-[#182030]/40 transition-colors">
                  <td className="py-3 px-4 font-medium text-white">
                    {item.component}
                  </td>
                  <td className="py-3 px-4 text-[#94a3b8] font-mono text-[11px]">
                    {item.baseline}
                  </td>
                  <td className="py-3 px-4 text-white font-mono text-[11px]">
                    {item.improved}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
