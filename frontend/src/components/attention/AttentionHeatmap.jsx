// src/components/attention/AttentionHeatmap.jsx
// D3-rendered attention weight heatmap.
// React owns the <svg> ref; D3 owns everything inside it.

import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

export default function AttentionHeatmap({ weights, tokens, selectedHead }) {
  const svgRef     = useRef(null)
  const tooltipRef = useRef(null)

  useEffect(() => {
    if (!weights || !tokens || selectedHead == null) return

    const headWeights = weights[selectedHead]   // [T][T]
    const T           = tokens.length
    if (!headWeights || T === 0) return

    // ── Dimensions ────────────────────────────────────────────────────
    const cellSize = Math.max(18, Math.min(48, Math.floor(480 / T)))
    const margin   = { top: 70, right: 20, bottom: 20, left: 80 }
    const width    = T * cellSize + margin.left + margin.right
    const height   = T * cellSize + margin.top  + margin.bottom

    // ── SVG setup ─────────────────────────────────────────────────────
    const svgEl = d3.select(svgRef.current)
    svgEl.selectAll('*').remove()  // Full redraw on each update

    svgEl
      .attr('width',  width)
      .attr('height', height)
      .style('overflow', 'visible')

    const g = svgEl.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // ── Colour scale: light (low weight) → indigo (high weight) ──────
    const colorScale = d3.scaleSequential()
      .domain([0, d3.max(headWeights.flat()) || 1])
      .interpolator(d3.interpolate('#1e1b4b', '#818cf8'))
      // dark-indigo → bright-indigo (matches our Tailwind indigo theme)

    // ── Flatten data for D3 join ──────────────────────────────────────
    const cells = headWeights.flatMap((row, i) =>
      row.map((value, j) => ({ i, j, value }))
    )

    // ── Draw cells ────────────────────────────────────────────────────
    const tooltip = d3.select(tooltipRef.current)

    g.selectAll('rect.cell')
      .data(cells)
      .join('rect')
      .attr('class', 'cell')
      .attr('x',      (d) => d.j * cellSize)
      .attr('y',      (d) => d.i * cellSize)
      .attr('width',  cellSize - 1)
      .attr('height', cellSize - 1)
      .attr('rx',     3)
      .attr('fill',   (d) => colorScale(d.value))
      .on('mouseover', (event, d) => {
        tooltip
          .style('opacity', 1)
          .style('left',    `${event.pageX + 12}px`)
          .style('top',     `${event.pageY - 28}px`)
          .html(
            `<span style="color:#a5b4fc">${tokens[d.i]}</span>` +
            ` → ` +
            `<span style="color:#a5b4fc">${tokens[d.j]}</span>` +
            `<br/>${d.value.toFixed(4)}`
          )
      })
      .on('mousemove', (event) => {
        tooltip
          .style('left', `${event.pageX + 12}px`)
          .style('top',  `${event.pageY - 28}px`)
      })
      .on('mouseout', () => tooltip.style('opacity', 0))

    // ── Row labels (query tokens — left axis) ─────────────────────────
    g.selectAll('text.row-label')
      .data(tokens)
      .join('text')
      .attr('class',              'row-label')
      .attr('x',                  -6)
      .attr('y',                  (_, i) => i * cellSize + cellSize / 2)
      .attr('text-anchor',        'end')
      .attr('dominant-baseline',  'middle')
      .attr('font-size',          Math.max(9, cellSize * 0.38))
      .attr('font-family',        'JetBrains Mono, monospace')
      .attr('fill',               '#9ca3af')
      .text((d) => d)

    // ── Column labels (key tokens — top axis) ─────────────────────────
    g.selectAll('text.col-label')
      .data(tokens)
      .join('text')
      .attr('class',          'col-label')
      .attr('x',              (_, i) => i * cellSize + cellSize / 2)
      .attr('y',              -8)
      .attr('text-anchor',    'middle')
      .attr('font-size',      Math.max(9, cellSize * 0.38))
      .attr('font-family',    'JetBrains Mono, monospace')
      .attr('fill',           '#9ca3af')
      .attr('transform',      (_, i) =>
        `rotate(-35, ${i * cellSize + cellSize / 2}, -8)`
      )
      .text((d) => d)

    // ── Axis labels ───────────────────────────────────────────────────
    svgEl.append('text')
      .attr('x',           margin.left / 2)
      .attr('y',           margin.top + (T * cellSize) / 2)
      .attr('text-anchor', 'middle')
      .attr('transform',
        `rotate(-90, ${margin.left / 2}, ${margin.top + (T * cellSize) / 2})`
      )
      .attr('font-size',   10)
      .attr('fill',        '#6b7280')
      .text('Query (from)')

    svgEl.append('text')
      .attr('x',           margin.left + (T * cellSize) / 2)
      .attr('y',           18)
      .attr('text-anchor', 'middle')
      .attr('font-size',   10)
      .attr('fill',        '#6b7280')
      .text('Key (to)')

  }, [weights, tokens, selectedHead])

  return (
    <div className="relative overflow-x-auto">
      <svg ref={svgRef} />
      {/* D3 tooltip — positioned fixed so it stays visible near the SVG */}
      <div
        ref       = {tooltipRef}
        className = "pointer-events-none fixed z-50 px-2 py-1 rounded text-xs \
                     font-mono bg-gray-900 border border-gray-700 text-gray-200 \
                     shadow-lg opacity-0 transition-opacity duration-75"
      />
    </div>
  )
}