"""
Interactive Plotly Visualizations for Airbnb Dublin Analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class AirbnbVisualizer:
    def __init__(self, analyzer):
        """Initialize with analyzer object"""
        self.analyzer = analyzer
        self.searches = analyzer.searches
        self.contacts = analyzer.contacts

    def create_executive_dashboard(self):
        """Main executive dashboards with KPIs"""
        kpis = self.analyzer.calculate_kpis()

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Conversion Funnel',
                'Searches vs Bookings Over Time',
                'Top Search Countries',
                'Acceptance Rate by Trip Length',
                'Response Time Distribution',
                'Check-in Day Patterns'
            ),
            specs=[
                [{'type': 'funnel'}, {'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'histogram'}, {'type': 'bar'}]
            ]
        )

        # 1. Conversion Funnel
        funnel_data = {
            'stage': ['Inquiries', 'Replied', 'Accepted', 'Booked'],
            'value': [
                kpis['total_inquiries'],
                kpis['total_replied'],
                kpis['total_accepted'],
                kpis['total_booked']
            ]
        }

        fig.add_trace(
            go.Funnel(
                y=funnel_data['stage'],
                x=funnel_data['value'],
                textinfo='value+percent initial',
                marker=dict(color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
            ),
            row=1, col=1
        )

        # 2. Time series
        searches_daily = self.searches.groupby('ds').size().reset_index(name='searches')
        bookings_daily = self.contacts[self.contacts['was_booked']].groupby(
            self.contacts['ts_booking_at'].dt.date
        ).size().reset_index(name='bookings')
        bookings_daily.columns = ['date', 'bookings']

        fig.add_trace(
            go.Scatter(
                x=searches_daily['ds'][:90],  # First 90 days
                y=searches_daily['searches'][:90],
                name='Searches',
                line=dict(color='#3498db', width=2)
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=bookings_daily['date'][:90],
                y=bookings_daily['bookings'][:90],
                name='Bookings',
                line=dict(color='#2ecc71', width=2)
            ),
            row=1, col=2
        )

        # 3. Top countries
        top_countries = self.searches['origin_country'].value_counts().head(10)

        fig.add_trace(
            go.Bar(
                x=top_countries.values,
                y=top_countries.index,
                orientation='h',
                marker_color='#9b59b6'
            ),
            row=1, col=3
        )

        # 4. Acceptance by nights
        nights_analysis = self.contacts.dropna(subset=['nights_bucket']).groupby(
            'nights_bucket'
        )['was_accepted'].mean() * 100

        fig.add_trace(
            go.Bar(
                x=nights_analysis.index.astype(str),
                y=nights_analysis.values,
                marker_color='#1abc9c',
                text=nights_analysis.values.round(1),
                texttemplate='%{text}%',
                textposition='outside'
            ),
            row=2, col=1
        )

        # 5. Response time histogram
        response_times = self.contacts['response_time_hours'].dropna()
        response_times = response_times[response_times <= 48]  # Filter outliers

        fig.add_trace(
            go.Histogram(
                x=response_times,
                nbinsx=30,
                marker_color='#e67e22'
            ),
            row=2, col=2
        )

        # 6. Check-in day patterns
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        checkin_searches = self.searches.dropna(subset=['checkin_dow']).groupby(
            'checkin_dow'
        ).size()

        fig.add_trace(
            go.Bar(
                x=[dow_names[int(i)] for i in checkin_searches.index if not pd.isna(i)],
                y=checkin_searches.values,
                marker_color='#34495e'
            ),
            row=2, col=3
        )

        # Update layout
        fig.update_layout(
            height=900,
            showlegend=False,
            title_text="<b>Airbnb Dublin Market Analysis Dashboard</b>",
            title_font_size=20
        )

        fig.write_html('dashboards/executive_dashboard.html')
        print("✓ Executive dashboards saved to: dashboards/executive_dashboard.html")
        return fig

    def create_gap_analysis_viz(self):
        """Supply-demand gap visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Search Volume vs Bookings by Trip Length',
                'Room Type Demand (Searches)',
                'Price Sensitivity Analysis',
                'Gap Score by Segment'
            ),
            specs=[
                [{'type': 'xy'}, {'type': 'domain'}],  # FIXED: domain for pie chart
                [{'type': 'xy'}, {'type': 'xy'}]
            ]
        )

        # 1. Nights gap
        demand_nights = self.searches.dropna(subset=['nights_bucket']).groupby(
            'nights_bucket'
        ).size()

        supply_nights = self.contacts[self.contacts['was_booked']].dropna(
            subset=['nights_bucket']
        ).groupby('nights_bucket').size()

        nights_categories = ['1-2 nights', '3-5 nights', '6-14 nights', '15+ nights']

        fig.add_trace(
            go.Bar(
                name='Searches',
                x=nights_categories,
                y=[demand_nights.get(c, 0) for c in nights_categories],
                marker_color='#3498db'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                name='Bookings',
                x=nights_categories,
                y=[supply_nights.get(c, 0) for c in nights_categories],
                marker_color='#2ecc71'
            ),
            row=1, col=1
        )

        # 2. Room type demand - FIXED for domain subplot
        room_demand = self.searches[
            self.searches['room_type_clean'] != 'No Filter'
            ]['room_type_clean'].value_counts()

        fig.add_trace(
            go.Pie(
                labels=room_demand.index,
                values=room_demand.values,
                hole=0.4,
                marker_colors=['#e74c3c', '#f39c12', '#9b59b6'],
                domain={'x': [0.55, 0.95], 'y': [0.55, 0.95]}  # Position manually
            ),
            row=1, col=2
        )

        # 3. Price sensitivity
        price_filtered = self.searches[self.searches['has_price_filter']].copy()

        if len(price_filtered) > 0:
            price_filtered['avg_price'] = (
                                                  price_filtered['filter_price_min'].fillna(0) +
                                                  price_filtered['filter_price_max'].fillna(200)
                                          ) / 2

            # Create bins and count
            price_filtered['price_bin'] = pd.cut(price_filtered['avg_price'], bins=10)
            price_demand = price_filtered.groupby('price_bin').size()

            # Get bin centers for x-axis
            bin_centers = [interval.mid for interval in price_demand.index]

            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=price_demand.values,
                    mode='lines+markers',
                    line=dict(color='#16a085', width=3),
                    marker=dict(size=8),
                    name='Search Volume'
                ),
                row=2, col=1
            )

        # 4. Gap score heatmap
        gap_data = self.searches.dropna(subset=['nights_bucket', 'origin_country']).groupby(
            ['nights_bucket', 'origin_country']
        ).size().reset_index(name='searches')

        top_countries = self.searches['origin_country'].value_counts().head(8).index
        gap_pivot = gap_data[gap_data['origin_country'].isin(top_countries)].pivot(
            index='nights_bucket',
            columns='origin_country',
            values='searches'
        ).fillna(0)

        if len(gap_pivot) > 0:
            fig.add_trace(
                go.Heatmap(
                    z=gap_pivot.values,
                    x=gap_pivot.columns,
                    y=gap_pivot.index.astype(str),
                    colorscale='YlOrRd',
                    text=gap_pivot.values,
                    texttemplate='%{text:.0f}',
                    textfont={"size": 10}
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            title_text="<b>Supply-Demand Gap Analysis</b>",
            showlegend=True
        )

        fig.write_html('dashboards/gap_analysis.html')
        print("✓ Gap analysis saved to: dashboards/gap_analysis.html")
        return fig

        # 3. Price sensitivity
        price_filtered = self.searches[self.searches['has_price_filter']].copy()
        price_filtered['avg_price'] = (
                                              price_filtered['filter_price_min'].fillna(0) +
                                              price_filtered['filter_price_max'].fillna(200)
                                      ) / 2

        price_bins = pd.cut(price_filtered['avg_price'], bins=10)
        price_demand = price_filtered.groupby(price_bins).size()

        fig.add_trace(
            go.Scatter(
                x=[str(b) for b in price_demand.index],
                y=price_demand.values,
                mode='lines+markers',
                line=dict(color='#16a085', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )

        # 4. Gap score heatmap
        gap_data = self.searches.dropna(subset=['nights_bucket', 'origin_country']).groupby(
            ['nights_bucket', 'origin_country']
        ).size().reset_index(name='searches')

        top_countries = self.searches['origin_country'].value_counts().head(8).index
        gap_pivot = gap_data[gap_data['origin_country'].isin(top_countries)].pivot(
            index='nights_bucket',
            columns='origin_country',
            values='searches'
        ).fillna(0)

        fig.add_trace(
            go.Heatmap(
                z=gap_pivot.values,
                x=gap_pivot.columns,
                y=gap_pivot.index.astype(str),
                colorscale='YlOrRd',
                text=gap_pivot.values,
                texttemplate='%{text:.0f}',
                textfont={"size": 10}
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="<b>Supply-Demand Gap Analysis</b>",
            showlegend=True
        )

        fig.write_html('dashboards/gap_analysis.html')
        print("✓ Gap analysis saved to: dashboards/gap_analysis.html")
        return fig

    def create_host_performance_viz(self):
        """Host behavior and performance visualizations"""
        host_metrics = self.analyzer.host_performance_metrics()
        active_hosts = host_metrics[host_metrics['total_inquiries'] >= 5]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Acceptance Rate Distribution',
                'Response Time vs Acceptance Rate',
                'Top vs Bottom Performers',
                'Message Count Impact'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ]
        )

        # 1. Acceptance rate distribution
        fig.add_trace(
            go.Histogram(
                x=active_hosts['acceptance_rate'] * 100,
                nbinsx=20,
                marker_color='#2ecc71',
                name='Hosts'
            ),
            row=1, col=1
        )

        # 2. Response time vs acceptance
        fig.add_trace(
            go.Scatter(
                x=active_hosts['median_response_hrs'],
                y=active_hosts['acceptance_rate'] * 100,
                mode='markers',
                marker=dict(
                    size=active_hosts['total_inquiries'] / 5,
                    color=active_hosts['booking_rate'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Booking<br>Rate")
                ),
                text=[f"Inquiries: {x}" for x in active_hosts['total_inquiries']],
                hovertemplate='<b>Response Time:</b> %{x:.1f}h<br>' +
                              '<b>Acceptance Rate:</b> %{y:.1f}%<br>' +
                              '%{text}',
                name='Hosts'
            ),
            row=1, col=2
        )

        # 3. Top vs bottom performers
        top_10 = active_hosts.nlargest(10, 'booking_rate')
        bottom_10 = active_hosts.nsmallest(10, 'booking_rate')

        fig.add_trace(
            go.Bar(
                name='Top 10',
                y=['Top'] * 10,
                x=top_10['booking_rate'] * 100,
                orientation='h',
                marker_color='#27ae60'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(
                name='Bottom 10',
                y=['Bottom'] * 10,
                x=bottom_10['booking_rate'] * 100,
                orientation='h',
                marker_color='#e74c3c'
            ),
            row=2, col=1
        )

        # 4. Messages impact
        fig.add_trace(
            go.Scatter(
                x=active_hosts['avg_messages'],
                y=active_hosts['booking_rate'] * 100,
                mode='markers',
                marker=dict(size=10, color='#9b59b6'),
                name='Hosts'
            ),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Acceptance Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="Response Time (hours)", row=1, col=2)
        fig.update_xaxes(title_text="Booking Rate (%)", row=2, col=1)
        fig.update_xaxes(title_text="Avg Messages", row=2, col=2)

        fig.update_yaxes(title_text="Acceptance Rate (%)", row=1, col=2)
        fig.update_yaxes(title_text="Booking Rate (%)", row=2, col=2)

        fig.update_layout(
            height=800,
            title_text="<b>Host Performance Analysis</b>",
            showlegend=True
        )

        fig.write_html('dashboards/host_performance.html')
        print("✓ Host performance viz saved to: dashboards/host_performance.html")
        return fig

    def create_temporal_analysis(self):
        """Time-based patterns and trends"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Search Volume Heatmap (Day of Week × Month)',
                'Lead Time Distribution',
                'Seasonal Booking Patterns',
                'Hour of Day Contact Patterns'
            ),
            specs=[
                [{'type': 'heatmap'}, {'type': 'box'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ]
        )

        # 1. DOW × Month heatmap
        searches_pivot = self.searches.groupby(
            ['search_dow', 'search_month']
        ).size().reset_index(name='count')

        heatmap_data = searches_pivot.pivot(
            index='search_dow',
            columns='search_month',
            values='count'
        ).fillna(0)

        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=[dow_names[i] for i in heatmap_data.index],
                colorscale='Blues',
                text=heatmap_data.values,
                texttemplate='%{text:.0f}'
            ),
            row=1, col=1
        )

        # 2. Lead time box plot
        lead_time_by_country = []
        top_countries = self.searches['origin_country'].value_counts().head(6).index

        for country in top_countries:
            country_data = self.searches[
                self.searches['origin_country'] == country
                ]['lead_time_days'].dropna()
            country_data = country_data[country_data <= 180]  # Filter outliers

            fig.add_trace(
                go.Box(
                    y=country_data,
                    name=country,
                    boxmean='sd'
                ),
                row=1, col=2
            )

        # 3. Seasonal patterns
        monthly_bookings = self.contacts[self.contacts['was_booked']].groupby(
            'checkin_month'
        ).size()

        monthly_searches = self.searches.groupby('checkin_month').size()

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig.add_trace(
            go.Scatter(
                x=month_names,
                y=[monthly_searches.get(i + 1, 0) for i in range(12)],
                name='Searches',
                line=dict(color='#3498db', width=3)
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=month_names,
                y=[monthly_bookings.get(i + 1, 0) for i in range(12)],
                name='Bookings',
                line=dict(color='#2ecc71', width=3)
            ),
            row=2, col=1
        )

        # 4. Hour of day patterns
        self.contacts['contact_hour'] = self.contacts['ts_contact_at'].dt.hour
        hourly_contacts = self.contacts.groupby('contact_hour').size()

        fig.add_trace(
            go.Bar(
                x=hourly_contacts.index,
                y=hourly_contacts.values,
                marker_color='#e67e22'
            ),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Month", row=1, col=1)
        fig.update_xaxes(title_text="Country", row=1, col=2)
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=2)

        fig.update_yaxes(title_text="Day of Week", row=1, col=1)
        fig.update_yaxes(title_text="Lead Time (days)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Contacts", row=2, col=2)

        fig.update_layout(
            height=800,
            title_text="<b>Temporal Patterns Analysis</b>",
            showlegend=True
        )

        fig.write_html('dashboards/temporal_analysis.html')
        print("✓ Temporal analysis saved to: dashboards/temporal_analysis.html")
        return fig

    def generate_all_dashboards(self):
        """Generate all visualization dashboards"""
        print("\n" + "=" * 50)
        print("GENERATING PLOTLY DASHBOARDS")
        print("=" * 50)

        self.create_executive_dashboard()
        self.create_gap_analysis_viz()
        self.create_host_performance_viz()
        self.create_temporal_analysis()

        print("\n✅ All dashboards generated successfully!")
        print("📁 Check the 'dashboards/' folder for HTML files")


# Usage
if __name__ == "__main__":
    from main_analysis import AirbnbDublinAnalyzer

    # Load data
    analyzer = AirbnbDublinAnalyzer(
        searches_path='data/searches.tsv',
        contacts_path='data/contacts.tsv'
    )
    analyzer.load_data()

    # Generate visualizations
    viz = AirbnbVisualizer(analyzer)
    viz.generate_all_dashboards()