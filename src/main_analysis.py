"""
Airbnb Dublin Market Analysis
Complete pipeline for demand/supply gap analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class AirbnbDublinAnalyzer:
    def __init__(self, searches_path, contacts_path):
        """Initialize with data paths"""
        self.searches_path = searches_path
        self.contacts_path = contacts_path
        self.searches = None
        self.contacts = None

    def load_data(self):
        """Load and perform initial data cleaning"""
        print("Loading data...")

        # Load searches
        self.searches = pd.read_csv(self.searches_path, sep='\t', low_memory=False)

        # Load contacts
        self.contacts = pd.read_csv(self.contacts_path, sep='\t', low_memory=False)

        print(f"Searches loaded: {len(self.searches):,} rows")
        print(f"Contacts loaded: {len(self.contacts):,} rows")

        # Clean searches data
        self._clean_searches()

        # Clean contacts data
        self._clean_contacts()

        return self

    def _clean_searches(self):
        """Clean and enrich searches dataset"""
        # Convert dates
        self.searches['ds'] = pd.to_datetime(self.searches['ds'])
        self.searches['ds_checkin'] = pd.to_datetime(self.searches['ds_checkin'])
        self.searches['ds_checkout'] = pd.to_datetime(self.searches['ds_checkout'])

        # Extract temporal features
        self.searches['search_year'] = self.searches['ds'].dt.year
        self.searches['search_month'] = self.searches['ds'].dt.month
        self.searches['search_dow'] = self.searches['ds'].dt.dayofweek
        self.searches['checkin_dow'] = self.searches['ds_checkin'].dt.dayofweek
        self.searches['checkin_week'] = self.searches['ds_checkin'].dt.isocalendar().week
        self.searches['checkin_month'] = self.searches['ds_checkin'].dt.month

        # Lead time (days between search and checkin)
        self.searches['lead_time_days'] = (
                self.searches['ds_checkin'] - self.searches['ds']
        ).dt.days

        # Parse room types (handle multiple comma-separated values)
        self.searches['room_type_clean'] = self.searches['filter_room_types'].apply(
            self._extract_primary_room_type
        )

        # Bucket nights into categories
        self.searches['nights_bucket'] = pd.cut(
            self.searches['n_nights'],
            bins=[0, 2, 5, 14, 999],
            labels=['1-2 nights', '3-5 nights', '6-14 nights', '15+ nights']
        )

        # Price range categories
        self.searches['has_price_filter'] = (
                self.searches['filter_price_min'].notna() |
                self.searches['filter_price_max'].notna()
        )

        print("✓ Searches cleaned and enriched")

    def _clean_contacts(self):
        """Clean and enrich contacts dataset"""
        # Convert timestamps
        time_cols = ['ts_contact_at', 'ts_reply_at', 'ts_accepted_at', 'ts_booking_at']
        for col in time_cols:
            self.contacts[col] = pd.to_datetime(self.contacts[col])

        # Convert dates
        self.contacts['ds_checkin'] = pd.to_datetime(self.contacts['ds_checkin'])
        self.contacts['ds_checkout'] = pd.to_datetime(self.contacts['ds_checkout'])

        # Calculate nights
        self.contacts['n_nights'] = (
                self.contacts['ds_checkout'] - self.contacts['ds_checkin']
        ).dt.days

        # Extract temporal features
        self.contacts['contact_dow'] = self.contacts['ts_contact_at'].dt.dayofweek
        self.contacts['checkin_dow'] = self.contacts['ds_checkin'].dt.dayofweek
        self.contacts['checkin_week'] = self.contacts['ds_checkin'].dt.isocalendar().week
        self.contacts['checkin_month'] = self.contacts['ds_checkin'].dt.month

        # Create conversion flags
        self.contacts['was_replied'] = self.contacts['ts_reply_at'].notna()
        self.contacts['was_accepted'] = self.contacts['ts_accepted_at'].notna()
        self.contacts['was_booked'] = self.contacts['ts_booking_at'].notna()

        # Response time (hours)
        self.contacts['response_time_hours'] = (
                (self.contacts['ts_reply_at'] - self.contacts['ts_contact_at'])
                .dt.total_seconds() / 3600
        )

        # Time to acceptance
        self.contacts['time_to_acceptance_hours'] = (
                (self.contacts['ts_accepted_at'] - self.contacts['ts_contact_at'])
                .dt.total_seconds() / 3600
        )

        # Nights bucket
        self.contacts['nights_bucket'] = pd.cut(
            self.contacts['n_nights'],
            bins=[0, 2, 5, 14, 999],
            labels=['1-2 nights', '3-5 nights', '6-14 nights', '15+ nights']
        )

        print("✓ Contacts cleaned and enriched")

    @staticmethod
    def _extract_primary_room_type(value):
        """Extract primary room type from comma-separated string"""
        if pd.isna(value) or value == '':
            return 'No Filter'

        # Split and get most common type
        types = [x.strip() for x in str(value).split(',') if x.strip()]
        if not types:
            return 'No Filter'

        # Return first non-empty
        for t in types:
            if t in ['Entire home/apt', 'Private room', 'Shared room']:
                return t
        return 'No Filter'

    def calculate_kpis(self):
        """Calculate key performance indicators"""
        print("\nCalculating KPIs...")

        kpis = {}

        # Overall metrics
        kpis['total_searches'] = len(self.searches)
        kpis['total_inquiries'] = len(self.contacts)
        kpis['total_replied'] = self.contacts['was_replied'].sum()
        kpis['total_accepted'] = self.contacts['was_accepted'].sum()
        kpis['total_booked'] = self.contacts['was_booked'].sum()

        # Conversion rates
        kpis['reply_rate'] = kpis['total_replied'] / kpis['total_inquiries']
        kpis['acceptance_rate'] = kpis['total_accepted'] / kpis['total_inquiries']
        kpis['booking_rate'] = kpis['total_booked'] / kpis['total_inquiries']

        # Average metrics
        kpis['avg_response_time_hours'] = self.contacts['response_time_hours'].mean()
        kpis['median_nights_searched'] = self.searches['n_nights'].median()
        kpis['median_nights_booked'] = self.contacts[
            self.contacts['was_booked']
        ]['n_nights'].median()

        print("\n=== KEY METRICS ===")
        print(f"Total Searches: {kpis['total_searches']:,}")
        print(f"Total Inquiries: {kpis['total_inquiries']:,}")
        print(f"Reply Rate: {kpis['reply_rate']:.1%}")
        print(f"Acceptance Rate: {kpis['acceptance_rate']:.1%}")
        print(f"Booking Rate: {kpis['booking_rate']:.1%}")
        print(f"Avg Response Time: {kpis['avg_response_time_hours']:.1f} hours")

        return kpis

    def demand_supply_gap_analysis(self):
        """Core gap analysis - compare search demand vs booking supply"""
        print("\n" + "=" * 50)
        print("DEMAND vs SUPPLY GAP ANALYSIS")
        print("=" * 50)

        # Aggregate demand by room type
        demand_by_room = self.searches[
            self.searches['room_type_clean'] != 'No Filter'
            ].groupby('room_type_clean').agg({
            'id_user': 'count',
            'n_nights': 'mean'
        }).rename(columns={'id_user': 'search_volume', 'n_nights': 'avg_nights_searched'})

        # Aggregate supply (bookings only)
        # Note: We don't have room_type in contacts, so we'll use acceptance as proxy
        supply_by_acceptance = self.contacts.groupby('was_accepted').agg({
            'id_guest': 'count',
            'n_nights': 'mean'
        })

        # Gap by nights bucket
        demand_by_nights = self.searches.dropna(subset=['nights_bucket']).groupby(
            'nights_bucket'
        ).size().rename('searches')

        bookings_by_nights = self.contacts[
            self.contacts['was_booked']
        ].dropna(subset=['nights_bucket']).groupby('nights_bucket').size().rename('bookings')

        gap_nights = pd.DataFrame({
            'searches': demand_by_nights,
            'bookings': bookings_by_nights
        }).fillna(0)

        gap_nights['gap'] = gap_nights['searches'] - gap_nights['bookings']
        gap_nights['gap_pct'] = (gap_nights['gap'] / gap_nights['searches'] * 100).round(1)
        gap_nights['booking_conversion'] = (
                gap_nights['bookings'] / gap_nights['searches'] * 100
        ).round(1)

        print("\n📊 GAP BY TRIP LENGTH:")
        print(gap_nights.sort_values('gap', ascending=False))

        # Gap by origin country (top 10)
        demand_by_country = self.searches.groupby('origin_country').size().rename('searches')

        # For supply, we need to infer - use inquiries as proxy
        supply_by_country = self.contacts[
            self.contacts['was_booked']
        ].groupby('id_guest').first()  # One row per guest

        top_countries = demand_by_country.nlargest(10)

        print("\n🌍 TOP 10 SEARCH COUNTRIES:")
        print(top_countries)

        return {
            'demand_by_room': demand_by_room,
            'gap_by_nights': gap_nights,
            'top_countries': top_countries
        }

    def analyze_by_dimensions(self):
        """Detailed analysis across multiple dimensions"""
        print("\n" + "=" * 50)
        print("DIMENSIONAL ANALYSIS")
        print("=" * 50)

        results = {}

        # 1. By Room Type
        room_analysis = self.searches[
            self.searches['room_type_clean'] != 'No Filter'
            ].groupby('room_type_clean').agg({
            'id_user': 'count',
            'n_nights': ['mean', 'median'],
            'n_guests_min': 'mean'
        }).round(2)

        print("\n🏠 ROOM TYPE ANALYSIS:")
        print(room_analysis)
        results['room_type'] = room_analysis

        # 2. By Nights Bucket
        nights_analysis = self.contacts.dropna(subset=['nights_bucket']).groupby(
            'nights_bucket'
        ).agg({
            'id_guest': 'count',
            'was_accepted': 'mean',
            'was_booked': 'mean',
            'response_time_hours': 'median'
        }).round(3)

        nights_analysis.columns = ['inquiries', 'acceptance_rate', 'booking_rate', 'median_response_hrs']

        print("\n📅 NIGHTS BUCKET ANALYSIS:")
        print(nights_analysis)
        results['nights'] = nights_analysis

        # 3. By Origin Country (top 10)
        country_searches = self.searches.groupby('origin_country').size().sort_values(ascending=False).head(10)

        print("\n🌐 TOP 10 COUNTRIES BY SEARCH VOLUME:")
        print(country_searches)
        results['countries'] = country_searches

        # 4. Check-in day of week patterns
        checkin_dow_searches = self.searches.dropna(subset=['checkin_dow']).groupby(
            'checkin_dow'
        ).size()

        checkin_dow_bookings = self.contacts[
            self.contacts['was_booked']
        ].groupby('checkin_dow').size()

        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        checkin_comparison = pd.DataFrame({
            'searches': checkin_dow_searches,
            'bookings': checkin_dow_bookings
        }).fillna(0)
        checkin_comparison.index = [dow_names[int(i)] if not pd.isna(i) else 'Unknown' for i in checkin_comparison.index]

        print("\n📆 CHECK-IN DAY OF WEEK:")
        print(checkin_comparison)
        results['checkin_dow'] = checkin_comparison

        return results

    def host_performance_metrics(self):
        """Analyze host behavior and performance"""
        print("\n" + "=" * 50)
        print("HOST PERFORMANCE ANALYSIS")
        print("=" * 50)

        host_metrics = self.contacts.groupby('id_host').agg({
            'id_guest': 'count',
            'was_replied': 'mean',
            'was_accepted': 'mean',
            'was_booked': 'mean',
            'response_time_hours': 'median',
            'n_messages': 'mean'
        }).round(3)

        host_metrics.columns = [
            'total_inquiries', 'reply_rate', 'acceptance_rate',
            'booking_rate', 'median_response_hrs', 'avg_messages'
        ]

        # Filter to active hosts (5+ inquiries)
        active_hosts = host_metrics[host_metrics['total_inquiries'] >= 5]

        print(f"\n👤 Active Hosts (5+ inquiries): {len(active_hosts)}")
        print(f"\n🏆 TOP PERFORMERS:")
        print(active_hosts.nlargest(10, 'booking_rate'))

        print(f"\n⚠️  LOW PERFORMERS:")
        print(active_hosts.nsmallest(10, 'acceptance_rate'))

        return host_metrics

    def generate_insights(self):
        """Generate actionable business insights"""
        print("\n" + "=" * 60)
        print("🎯 KEY INSIGHTS & RECOMMENDATIONS")
        print("=" * 60)

        insights = []

        # Insight 1: Conversion funnel
        total_inquiries = len(self.contacts)
        replied = self.contacts['was_replied'].sum()
        accepted = self.contacts['was_accepted'].sum()
        booked = self.contacts['was_booked'].sum()

        print(f"\n1️⃣  CONVERSION FUNNEL LEAK:")
        print(f"   Inquiries → Replied: {replied / total_inquiries:.1%} ({total_inquiries - replied:,} lost)")
        print(f"   Replied → Accepted: {accepted / replied:.1%} ({replied - accepted:,} lost)")
        print(f"   Accepted → Booked: {booked / accepted:.1%} ({accepted - booked:,} lost)")
        insights.append("Biggest drop-off is at reply stage - need to improve host responsiveness")

        # Insight 2: Response time impact
        fast_response = self.contacts[self.contacts['response_time_hours'] <= 1]
        slow_response = self.contacts[self.contacts['response_time_hours'] > 24]

        if len(fast_response) > 0 and len(slow_response) > 0:
            fast_accept_rate = fast_response['was_accepted'].mean()
            slow_accept_rate = slow_response['was_accepted'].mean()

            print(f"\n2️⃣  RESPONSE TIME IMPACT:")
            print(f"   <1 hour response: {fast_accept_rate:.1%} acceptance rate")
            print(f"   >24 hour response: {slow_accept_rate:.1%} acceptance rate")
            print(f"   Impact: {(fast_accept_rate - slow_accept_rate) * 100:.1f}pp difference")
            insights.append("Fast responses significantly improve acceptance rates")

        # Insight 3: Trip length preferences
        search_nights_median = self.searches['n_nights'].median()
        booked_nights_median = self.contacts[self.contacts['was_booked']]['n_nights'].median()

        print(f"\n3️⃣  TRIP LENGTH MISMATCH:")
        print(f"   Guests search for: {search_nights_median} nights (median)")
        print(f"   Actual bookings: {booked_nights_median} nights (median)")

        # Insight 4: International vs domestic
        domestic_searches = (self.searches['origin_country'] == 'IE').sum()
        international_searches = len(self.searches) - domestic_searches

        print(f"\n4️⃣  MARKET COMPOSITION:")
        print(f"   Domestic (IE): {domestic_searches:,} ({domestic_searches / len(self.searches):.1%})")
        print(f"   International: {international_searches:,} ({international_searches / len(self.searches):.1%})")

        return insights


# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AirbnbDublinAnalyzer(
        searches_path='data/searches.tsv',
        contacts_path='data/contacts.tsv'
    )

    # Run full pipeline
    analyzer.load_data()
    kpis = analyzer.calculate_kpis()
    gap_analysis = analyzer.demand_supply_gap_analysis()
    dimensional_analysis = analyzer.analyze_by_dimensions()
    host_metrics = analyzer.host_performance_metrics()
    insights = analyzer.generate_insights()

    print("\n" + "=" * 60)
    print("✅ Analysis complete! Ready for visualization.")
    print("=" * 60)