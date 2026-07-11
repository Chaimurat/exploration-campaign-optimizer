"""
well_gantt.py
-------------
Interactive Plotly Gantt chart for well workover schedules.
Y-axis: Rigs   X-axis: Days   Colour: Well
Mobilisation gaps appear as blank space between bars on each rig.
"""

import os
import plotly.graph_objects as go
import plotly.express as px


def generate_well_gantt(schedule, instance,
                        output_html_path="images/gantt_wellworkover.html",
                        title="Well Workover Schedule -- Rig Allocation"):

    # Sort by rig then start time
    sched = sorted(schedule, key=lambda x: (x['rig'], x['start']))

    # Assign a distinct colour per well
    well_names = sorted(set(s['well_name'] for s in sched))
    palette    = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    colour_map = {name: palette[i % len(palette)]
                  for i, name in enumerate(well_names)}

    fig = go.Figure()

    for entry in sched:
        rig_label  = entry['rig_name']
        well_label = entry['well_name']
        start      = entry['start']
        end        = entry['end']
        flow_rate  = entry['flow_rate']
        duration   = entry['duration']

        hover = (
            "<b>{}</b> on {}<br>"
            "Start: day {}<br>"
            "End:   day {}<br>"
            "Duration: {} days<br>"
            "Flow rate: {:,} bbl/day"
        ).format(well_label, rig_label, start, end, duration, flow_rate)

        fig.add_trace(go.Bar(
            name        = well_label,
            y           = [rig_label],
            x           = [duration],
            base        = [start],
            orientation = 'h',
            marker_color= colour_map[well_label],
            hovertemplate = hover + "<extra></extra>",
            showlegend  = well_label not in [t.name for t in fig.data[:-1]]
        ))

    fig.update_layout(
        title       = dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title = "Campaign Day",
        yaxis_title = "Rig",
        barmode     = 'stack',
        yaxis       = dict(categoryorder='category ascending'),
        legend_title= "Well",
        height      = 400,
        plot_bgcolor= 'white',
        paper_bgcolor='white',
        xaxis       = dict(gridcolor='#eeeeee', zeroline=False),
    )

    # Add mobilisation gap annotations
    rig_groups = {}
    for entry in sched:
        rig_groups.setdefault(entry['rig_name'], []).append(entry)

    for rig_name, entries in rig_groups.items():
        entries_sorted = sorted(entries, key=lambda x: x['start'])
        for i in range(len(entries_sorted) - 1):
            gap_start = entries_sorted[i]['end']
            gap_end   = entries_sorted[i+1]['start']
            if gap_end > gap_start:
                mid = (gap_start + gap_end) / 2
                fig.add_annotation(
                    x=mid, y=rig_name,
                    text="mob<br>{}d".format(int(gap_end - gap_start)),
                    showarrow=False,
                    font=dict(size=9, color='#888888'),
                    bgcolor='rgba(255,255,255,0.7)'
                )

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    fig.write_html(output_html_path)
    print("[SUCCESS] Gantt chart saved to: {}".format(output_html_path))
