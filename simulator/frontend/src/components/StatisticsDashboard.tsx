import React, { useState, useMemo } from 'react';
import {
  Search,
  FileDown,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  X,
  Database,
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

type MacroMode = 'fold0' | 'fivefold' | 'folds';
type SortField = 'species' | 'enhanced_map50' | 'delta_vs_paper' | 'delta_vs_recreation';

const DeltaChip: React.FC<{ value: number; unit?: string; signed?: boolean }> = ({
  value,
  unit = '%',
  signed = true,
}) => (
  <span className={`delta-chip ${value >= 0 ? 'delta-up' : 'delta-down'}`}>
    {value > 0 && signed ? '+' : ''}
    {value.toFixed(2).replace(/\.00$/, '.0')}
    {unit}
  </span>
);

const SectionCaption: React.FC<{
  index: string;
  title: string;
  description: string;
  children?: React.ReactNode;
}> = ({ index, title, description, children }) => (
  <div className="section-caption flex-wrap gap-x-4 gap-y-3">
    <span className="eyebrow numeric">{index}</span>
    <h2 className="font-display text-[25px] leading-none font-medium tracking-[-0.02em] text-ink">
      {title}
    </h2>
    <p className="hidden text-[12px] text-ink-3 lg:block">{description}</p>
    {children && <div className="ml-auto">{children}</div>}
  </div>
);

const FoldTable: React.FC<{ title: string; qualifier: string; rows: FoldDetail[] }> = ({
  title,
  qualifier,
  rows,
}) => (
  <div>
    <div className="mb-2 flex items-baseline justify-between gap-3">
      <h3 className="text-[12px] font-semibold text-ink">{title}</h3>
      <span className="eyebrow">{qualifier}</span>
    </div>
    <div className="overflow-x-auto">
      <table className="w-full text-left text-[11px]">
        <thead>
          <tr className="border-b border-line">
            {['Fold', 'P', 'R', 'mAP@50', 'mAP@50-95', 'F1'].map((h, i) => (
              <th
                key={h}
                scope="col"
                className={`eyebrow py-2 font-medium ${i === 0 ? 'pr-2' : 'px-2 text-right'}`}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="numeric divide-y divide-line/70">
          {rows.map((f) => (
            <tr
              key={f.fold}
              className={`transition-colors hover:bg-surface-2/70 ${
                f.is_active_checkpoint ? 'bg-leaf-soft/50' : ''
              }`}
            >
              <td className="py-2 pr-2 text-ink">
                <span className="flex items-center gap-1.5">
                  Fold {f.fold}
                  {f.is_active_checkpoint && (
                    <span className="rounded-[2px] bg-ember-soft px-1 py-0.5 text-[9px] font-medium text-ember">
                      ACTIVE
                    </span>
                  )}
                </span>
              </td>
              <td className="px-2 py-2 text-right text-ink-2">{f.precision.toFixed(1)}</td>
              <td className="px-2 py-2 text-right text-ink-2">{f.recall.toFixed(1)}</td>
              <td className="px-2 py-2 text-right font-medium text-ink">{f.map50.toFixed(2)}%</td>
              <td className="px-2 py-2 text-right text-ink-2">{f.map50_95.toFixed(2)}%</td>
              <td className="px-2 py-2 text-right text-ink-2">{f.f1.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

export const StatisticsDashboard: React.FC<StatisticsDashboardProps> = ({ stats }) => {
  const [macroMode, setMacroMode] = useState<MacroMode>('fold0');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedOrder, setSelectedOrder] = useState<string>('all');
  const [sortBy, setSortBy] = useState<SortField>('enhanced_map50');
  const [sortAsc, setSortAsc] = useState(false);

  const orders = useMemo(() => {
    if (!stats) return ['all'];
    return ['all', ...Array.from(new Set(stats.species.map((s) => s.order)))];
  }, [stats]);

  const filteredSpecies = useMemo(() => {
    if (!stats) return [];
    return stats.species
      .filter((s) => {
        const q = searchTerm.trim().toLowerCase();
        const matchesSearch =
          !q || s.species.toLowerCase().includes(q) || s.order.toLowerCase().includes(q);
        const matchesOrder =
          selectedOrder === 'all' || s.order.toLowerCase() === selectedOrder.toLowerCase();
        return matchesSearch && matchesOrder;
      })
      .sort((a, b) => {
        const va = a[sortBy];
        const vb = b[sortBy];
        const cmp =
          typeof va === 'string' && typeof vb === 'string'
            ? va.localeCompare(vb)
            : (va as number) - (vb as number);
        return sortAsc ? cmp : -cmp;
      });
  }, [stats, searchTerm, selectedOrder, sortBy, sortAsc]);

  const toggleSort = (field: SortField) => {
    if (sortBy === field) setSortAsc((v) => !v);
    else {
      setSortBy(field);
      setSortAsc(false);
    }
  };

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
    const csv =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const link = document.createElement('a');
    link.href = encodeURI(csv);
    link.download = 'PaddyYOLO_Species_Benchmark.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!stats) {
    return (
      <div className="panel mt-8 flex flex-col items-center gap-3 px-6 py-16 text-center">
        <Database className="h-6 w-6 text-ink-3" />
        <h2 className="font-display text-[24px] text-ink">Benchmark data unavailable</h2>
        <p className="max-w-sm text-[13px] leading-relaxed text-ink-2">
          Results are served by the evaluation backend. Start the API on 127.0.0.1:8000 to load the
          published comparison tables.
        </p>
      </div>
    );
  }

  const primary =
    stats.macro.fold0_rows.find((r) => /^mAP@50 \(/.test(r.metric)) ?? stats.macro.fold0_rows[0];
  const secondary = stats.macro.fold0_rows.filter((r) => r !== primary).slice(0, 3);
  const fiveFoldPrimary = stats.macro.five_fold_rows[0];
  const comparison = [
    { label: 'Published paper', value: primary.paper, color: 'var(--c-ink-3)' },
    { label: 'Baseline recreation', value: primary.recreation, color: 'var(--c-baseline)' },
    { label: 'Enhanced architecture', value: primary.enhanced, color: 'var(--c-leaf)' },
  ];
  const maxComparison = Math.max(...comparison.map((c) => c.value), 1);

  const macroModes: { id: MacroMode; label: string }[] = [
    { id: 'fold0', label: 'Fold 0 · active checkpoint' },
    { id: 'fivefold', label: '5-fold · mean ± σ' },
    { id: 'folds', label: 'Individual folds' },
  ];

  const sortHeader = (field: SortField, label: string, align: 'left' | 'right' = 'right') => (
    <th
      scope="col"
      aria-sort={sortBy === field ? (sortAsc ? 'ascending' : 'descending') : 'none'}
      className={`px-3 py-3 font-medium ${align === 'right' ? 'text-right' : 'text-left'}`}
    >
      <button
        type="button"
        onClick={() => toggleSort(field)}
        className={`eyebrow inline-flex items-center gap-1.5 transition-colors hover:text-ink ${
          sortBy === field ? 'text-ink' : ''
        }`}
      >
        <span>{label}</span>
        {sortBy === field ? (
          sortAsc ? (
            <ArrowUp className="h-3 w-3" />
          ) : (
            <ArrowDown className="h-3 w-3" />
          )
        ) : (
          <ArrowUpDown className="h-3 w-3 opacity-50" />
        )}
        {align === 'right' && <span className="sr-only">sort</span>}
      </button>
    </th>
  );

  return (
    <div className="space-y-9 pt-7">
      {/* Editorial page header */}
      <header className="animate-rise">
        <div className="flex items-center gap-3">
          <span className="eyebrow eyebrow-ink">Workspace 02 / published results</span>
          <span className="h-px w-8 bg-line-strong" />
          <span className="eyebrow">Reproducible audit</span>
        </div>
        <h2 className="mt-2 font-display text-[clamp(34px,4.4vw,52px)] leading-[1.03] font-medium tracking-[-0.03em] text-ink">
          Benchmark evidence
        </h2>
        <p className="mt-2 max-w-[680px] text-[13px] leading-relaxed text-ink-2">
          A reproducible comparison of the published Paddy-YOLO results, the baseline recreation, and
          the enhanced EMA + DySample architecture across the full rice pest dataset.
        </p>
      </header>

      {/* Primary endpoint — display number plus a three-way comparison */}
      <section className="animate-rise-2 grid grid-cols-1 gap-8 border-y border-line py-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)] lg:gap-14">
        <div>
          <span className="eyebrow eyebrow-ink">Primary endpoint / mAP@50 · fold 0</span>
          <div className="numeric mt-3 flex items-baseline gap-1 font-display leading-[0.86] font-medium tracking-[-0.045em] text-ink">
            <span className="text-[clamp(64px,9vw,108px)]">{primary.enhanced.toFixed(2)}</span>
            <span className="text-[clamp(28px,3.6vw,44px)] text-ink-2">%</span>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <DeltaChip value={primary.delta_vs_paper} />
            <span className="text-[11px] text-ink-3">vs published paper</span>
            <span className="mx-1 h-3 w-px bg-line" />
            <DeltaChip value={primary.delta_vs_recreation} />
            <span className="text-[11px] text-ink-3">vs baseline recreation</span>
          </div>
          <p className="mt-5 max-w-[420px] text-[12px] leading-relaxed text-ink-2">
            Measured on the active checkpoint of the first holdout fold. Five-fold cross-validation
            across all 2,024 images reports{' '}
            <span className="numeric font-medium text-ink">{fiveFoldPrimary?.enhanced ?? '—'}</span> with
            substantially tighter variance.
          </p>
        </div>

        <div className="flex flex-col justify-center">
          <span className="eyebrow">mAP@50 · three-way comparison</span>
          <div className="mt-5 space-y-4">
            {comparison.map((c) => (
              <div key={c.label}>
                <div className="mb-1.5 flex items-baseline justify-between gap-3">
                  <span className="text-[12px] text-ink-2">{c.label}</span>
                  <span className="numeric text-[13px] font-medium text-ink">
                    {c.value.toFixed(2)}%
                  </span>
                </div>
                <div className="h-[7px] w-full overflow-hidden rounded-full bg-surface-2">
                  <div
                    className="animate-bar h-full origin-left rounded-full opacity-90"
                    style={{ width: `${(c.value / maxComparison) * 100}%`, backgroundColor: c.color }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="eyebrow mt-5">Bars scaled to the strongest result</p>
        </div>
      </section>

      {/* Supporting endpoints — a ruled row instead of four identical cards */}
      <section className="animate-rise-3 grid grid-cols-1 border-b border-line pb-7 sm:grid-cols-3 sm:divide-x sm:divide-line">
        {secondary.map((row) => (
          <div key={row.metric} className="border-b border-line py-4 sm:border-b-0 sm:px-6 sm:py-0 sm:first:pl-0">
            <span className="eyebrow">{row.metric}</span>
            <div className="numeric mt-2 font-display text-[32px] leading-none font-medium tracking-[-0.03em] text-ink">
              {row.enhanced.toFixed(2)}
              <span className="ml-0.5 text-[16px] text-ink-3">{row.unit}</span>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <DeltaChip value={row.delta_vs_recreation} unit={row.unit} />
              <span className="text-[11px] text-ink-3">vs recreation</span>
            </div>
          </div>
        ))}
      </section>

      {/* 01 · Macro benchmark */}
      <section>
        <SectionCaption
          index="01"
          title="Macro benchmark"
          description="Published paper vs baseline recreation vs enhanced architecture"
        >
          <div className="flex flex-wrap items-center gap-4">
            {macroModes.map((m) => (
              <button
                key={m.id}
                type="button"
                data-active={macroMode === m.id}
                onClick={() => setMacroMode(m.id)}
                className="tab-link py-1.5 text-[12px]"
              >
                {m.label}
              </button>
            ))}
          </div>
        </SectionCaption>

        {macroMode === 'folds' ? (
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            <FoldTable
              title="Baseline recreation"
              qualifier="Paddy YOLO"
              rows={stats.macro.folds.recreation}
            />
            <FoldTable
              title="Enhanced architecture"
              qualifier="EMA + DySample"
              rows={stats.macro.folds.enhanced}
            />
          </div>
        ) : (
          <div className="panel overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] text-left text-[12px]">
                <thead className="bg-surface-2">
                  <tr className="border-b border-line">
                    <th scope="col" className="eyebrow px-4 py-3 font-medium">Metric</th>
                    <th scope="col" className="eyebrow px-3 py-3 text-right font-medium">Paper</th>
                    <th scope="col" className="eyebrow px-3 py-3 text-right font-medium">Recreation</th>
                    <th scope="col" className="eyebrow eyebrow-leaf bg-leaf-soft/50 px-3 py-3 text-right font-medium">
                      Enhanced
                    </th>
                    <th scope="col" className="eyebrow px-3 py-3 text-right font-medium">Δ vs paper</th>
                    <th scope="col" className="eyebrow px-4 py-3 text-right font-medium">Δ vs rec</th>
                  </tr>
                </thead>
                <tbody className="numeric divide-y divide-line/70">
                  {macroMode === 'fold0'
                    ? stats.macro.fold0_rows.map((row: MacroFold0Row) => (
                        <tr key={row.metric} className="transition-colors hover:bg-surface-2/60">
                          <th scope="row" className="px-4 py-3 text-left font-sans text-[12px] font-medium text-ink">
                            {row.metric}
                          </th>
                          <td className="px-3 py-3 text-right text-ink-2">
                            {row.paper.toFixed(2)}{row.unit}
                          </td>
                          <td className="px-3 py-3 text-right text-ink-2">
                            {row.recreation.toFixed(2)}{row.unit}
                          </td>
                          <td className="bg-leaf-soft/40 px-3 py-3 text-right text-[14px] font-medium text-ink">
                            {row.enhanced.toFixed(2)}{row.unit}
                          </td>
                          <td className="px-3 py-3 text-right">
                            <DeltaChip value={row.delta_vs_paper} unit={row.unit} />
                          </td>
                          <td className="px-4 py-3 text-right">
                            <DeltaChip value={row.delta_vs_recreation} unit={row.unit} />
                          </td>
                        </tr>
                      ))
                    : stats.macro.five_fold_rows.map((row: MacroFiveFoldRow) => (
                        <tr key={row.metric} className="transition-colors hover:bg-surface-2/60">
                          <th scope="row" className="px-4 py-3 text-left font-sans text-[12px] font-medium text-ink">
                            {row.metric}
                          </th>
                          <td className="px-3 py-3 text-right text-ink-2">{row.paper}</td>
                          <td className="px-3 py-3 text-right text-ink-2">{row.recreation}</td>
                          <td className="bg-leaf-soft/40 px-3 py-3 text-right text-[14px] font-medium text-ink">
                            {row.enhanced}
                          </td>
                          <td className="px-3 py-3 text-right">
                            <span className="delta-chip delta-up">{row.delta_vs_paper}</span>
                          </td>
                          <td className="px-4 py-3 text-right">
                            <span className="delta-chip delta-up">{row.delta_vs_recreation}</span>
                          </td>
                        </tr>
                      ))}
                </tbody>
              </table>
            </div>
            <div className="flex items-center justify-between border-t border-line bg-surface-2/50 px-4 py-2.5">
              <span className="eyebrow">
                {macroMode === 'fold0'
                  ? 'Table 8 audit · fold 0 active checkpoint'
                  : 'Five-partition holdout · mean ± standard deviation'}
              </span>
              <span className="eyebrow">{stats.macro.fold0_rows.length} endpoints</span>
            </div>
          </div>
        )}
      </section>

      {/* 02 · Species-level benchmark */}
      <section>
        <SectionCaption
          index="02"
          title="Species-level results"
          description="Per-class detection accuracy across 16 rice pest species"
        >
          <button
            type="button"
            onClick={handleExportCSV}
            className="ctl h-9 gap-2 px-3 text-[11px] font-medium"
            title="Export the complete species dataset as CSV"
          >
            <FileDown className="h-3.5 w-3.5" />
            Export CSV
          </button>
        </SectionCaption>

        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative">
              <label htmlFor="species-search" className="sr-only">Search species</label>
              <Search className="absolute top-2.5 left-2.5 h-3.5 w-3.5 text-ink-3" />
              <input
                id="species-search"
                type="search"
                placeholder="Search species or order…"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="h-9 w-[220px] rounded-[4px] border border-line bg-surface pr-3 pl-8 text-[12px] text-ink placeholder:text-ink-3 focus:border-leaf focus:outline-none"
              />
            </div>
            <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Filter by taxonomic order">
              {orders.map((ord) => (
                <button
                  key={ord}
                  type="button"
                  onClick={() => setSelectedOrder(ord)}
                  className={`ctl h-8 px-2.5 text-[11px] capitalize ${
                    selectedOrder === ord ? 'font-semibold' : ''
                  }`}
                  data-on={selectedOrder === ord}
                >
                  {ord === 'all' ? 'All orders' : ord}
                </button>
              ))}
            </div>
          </div>
          <span className="eyebrow numeric">
            {filteredSpecies.length} / {stats.species.length} species shown
          </span>
        </div>

        <div className="panel overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[980px] text-left text-[12px]">
              <thead className="bg-surface-2">
                <tr className="border-b border-line">
                  {sortHeader('species', 'Species', 'left')}
                  <th scope="col" className="eyebrow px-3 py-3 font-medium">Order</th>
                  <th scope="col" className="eyebrow px-3 py-3 text-right font-medium">Paper</th>
                  <th scope="col" className="eyebrow px-3 py-3 text-right font-medium">Recreation</th>
                  {sortHeader('enhanced_map50', 'Enhanced mAP@50')}
                  <th scope="col" className="eyebrow px-3 py-3 text-right font-medium">P / R</th>
                  <th scope="col" className="eyebrow px-3 py-3 text-right font-medium">mAP@50-95</th>
                  {sortHeader('delta_vs_paper', 'Δ vs paper')}
                  {sortHeader('delta_vs_recreation', 'Δ vs rec')}
                  <th scope="col" className="eyebrow px-4 py-3 font-medium">Relative gain</th>
                </tr>
              </thead>
              <tbody className="numeric divide-y divide-line/70">
                {filteredSpecies.map((item: SpeciesItem) => {
                  const gainPct = Math.min(100, Math.max(0, (item.delta_vs_recreation / 20) * 100));
                  return (
                    <tr key={item.class_id} className="transition-colors hover:bg-surface-2/60">
                      <th scope="row" className="px-3 py-2.5 text-left font-sans text-[12px] font-medium text-ink capitalize">
                        {item.species}
                      </th>
                      <td className="px-3 py-2.5 font-sans text-[11px] text-ink-3">{item.order}</td>
                      <td className="px-3 py-2.5 text-right text-ink-2">{item.paper_map50.toFixed(1)}%</td>
                      <td className="px-3 py-2.5 text-right text-ink-2">{item.recreation_map50.toFixed(1)}%</td>
                      <td className="bg-leaf-soft/40 px-3 py-2.5 text-right text-[13px] font-medium text-ink">
                        {item.enhanced_map50.toFixed(1)}%
                        <span className="ml-1 text-[10px] font-normal text-ink-3">
                          ±{item.enhanced_map50_std.toFixed(1)}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 text-right text-[11px] text-ink-2">
                        {item.enhanced_p.toFixed(0)} / {item.enhanced_r.toFixed(0)}
                      </td>
                      <td className="px-3 py-2.5 text-right text-ink-2">{item.enhanced_map50_95.toFixed(1)}%</td>
                      <td className="px-3 py-2.5 text-right">
                        <DeltaChip value={item.delta_vs_paper} />
                      </td>
                      <td className="px-3 py-2.5 text-right">
                        <DeltaChip value={item.delta_vs_recreation} />
                      </td>
                      <td className="min-w-[110px] px-4 py-2.5">
                        <div className="h-[4px] w-full overflow-hidden rounded-full bg-line">
                          <div
                            className="h-full rounded-full bg-leaf transition-all duration-500"
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

          {filteredSpecies.length === 0 && (
            <div className="flex flex-col items-center gap-2 px-6 py-10 text-center">
              <X className="h-4 w-4 text-ink-3" />
              <p className="text-[13px] text-ink-2">No species match the current filters.</p>
              <button
                type="button"
                onClick={() => {
                  setSearchTerm('');
                  setSelectedOrder('all');
                }}
                className="ctl mt-1 h-8 px-3 text-[11px] font-medium"
              >
                Clear filters
              </button>
            </div>
          )}

          <div className="flex items-center justify-between border-t border-line bg-surface-2/50 px-4 py-2.5">
            <span className="eyebrow">mAP@50 shown as mean ± standard deviation over five folds</span>
            <span className="eyebrow">Click a column heading to sort</span>
          </div>
        </div>
      </section>

      {/* 03 · Architecture — definition list, no nested cards */}
      <section>
        <SectionCaption
          index="03"
          title="Architecture & ablation"
          description="Structural differences between the two evaluated models"
        />
        <dl className="grid grid-cols-1 gap-x-12 lg:grid-cols-2">
          {stats.architecture.map((item: ArchitectureSpec) => (
            <div
              key={item.component}
              className="grid grid-cols-1 gap-3 border-b border-line py-4 sm:grid-cols-[150px_minmax(0,1fr)] sm:gap-5"
            >
              <dt className="eyebrow eyebrow-ink pt-1">{item.component}</dt>
              <dd className="grid grid-cols-2 gap-4">
                <div>
                  <span className="eyebrow">Baseline</span>
                  <p className="numeric mt-1 text-[12px] leading-relaxed text-ink-2">{item.baseline}</p>
                </div>
                <div>
                  <span className="eyebrow">Enhanced</span>
                  <p className="numeric mt-1 text-[12px] leading-relaxed font-medium text-ink">
                    {item.improved}
                  </p>
                </div>
              </dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  );
};
