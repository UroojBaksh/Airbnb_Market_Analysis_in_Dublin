"""
Export processed data for Looker Studio
"""

import pandas as pd
import sys
sys.path.insert(0, 'src')
from main_analysis import AirbnbDublinAnalyzer

print("="*60)
print(" EXPORTING DATA FOR LOOKER STUDIO ".center(60))
print("="*60)

# Load and process data
analyzer = AirbnbDublinAnalyzer(
    searches_path='data/searches.tsv',
    contacts_path='data/contacts.tsv'
)
analyzer.load_data()

# Export 1: Searches Summary
searches_export = analyzer.searches[[
    'ds', 'ds_checkin', 'ds_checkout',
    'n_nights', 'n_guests_min', 'n_guests_max',
    'origin_country', 'room_type_clean', 'nights_bucket',
    'search_year', 'search_month', 'checkin_dow', 'checkin_month',
    'lead_time_days', 'has_price_filter'
]].copy()

# Convert categorical to string
searches_export['room_type_clean'] = searches_export['room_type_clean'].astype(str)
searches_export['nights_bucket'] = searches_export['nights_bucket'].astype(str)

# Convert date columns to string for Looker
searches_export['ds'] = searches_export['ds'].dt.strftime('%Y-%m-%d')
searches_export['ds_checkin'] = searches_export['ds_checkin'].dt.strftime('%Y-%m-%d')
searches_export['ds_checkout'] = searches_export['ds_checkout'].dt.strftime('%Y-%m-%d')

searches_export.to_csv('output/looker_searches.csv', index=False)
print(f"✓ Searches exported: {len(searches_export):,} rows")

# Export 2: Contacts/Bookings Summary
contacts_export = analyzer.contacts[[
    'id_guest', 'id_host', 'id_listing',
    'ts_contact_at', 'ts_reply_at', 'ts_accepted_at', 'ts_booking_at',
    'ds_checkin', 'ds_checkout', 'n_nights', 'n_guests', 'n_messages',
    'was_replied', 'was_accepted', 'was_booked',
    'response_time_hours', 'checkin_dow', 'checkin_month', 'nights_bucket'
]].copy()

# Convert categorical to string
contacts_export['nights_bucket'] = contacts_export['nights_bucket'].astype(str)

# Convert timestamps to strings
contacts_export['ts_contact_at'] = contacts_export['ts_contact_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
contacts_export['ts_reply_at'] = contacts_export['ts_reply_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
contacts_export['ts_accepted_at'] = contacts_export['ts_accepted_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
contacts_export['ts_booking_at'] = contacts_export['ts_booking_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
contacts_export['ds_checkin'] = contacts_export['ds_checkin'].dt.strftime('%Y-%m-%d')
contacts_export['ds_checkout'] = contacts_export['ds_checkout'].dt.strftime('%Y-%m-%d')

contacts_export.to_csv('output/looker_contacts.csv', index=False)
print(f"✓ Contacts exported: {len(contacts_export):,} rows")

# Export 3: Gap Analysis Summary (Pre-aggregated for performance)
gap_by_nights = analyzer.searches.dropna(subset=['nights_bucket']).groupby('nights_bucket').size().reset_index(name='searches')
bookings_by_nights = analyzer.contacts[analyzer.contacts['was_booked']].dropna(subset=['nights_bucket']).groupby('nights_bucket').size().reset_index(name='bookings')

# Convert categorical to string before merge
gap_by_nights['nights_bucket'] = gap_by_nights['nights_bucket'].astype(str)
bookings_by_nights['nights_bucket'] = bookings_by_nights['nights_bucket'].astype(str)

# Merge and fill
gap_summary = gap_by_nights.merge(bookings_by_nights, on='nights_bucket', how='left')
gap_summary['bookings'] = gap_summary['bookings'].fillna(0).astype(int)
gap_summary['gap'] = (gap_summary['searches'] - gap_summary['bookings']).astype(int)
gap_summary['conversion_rate'] = (gap_summary['bookings'] / gap_summary['searches'] * 100).round(1)

gap_summary.to_csv('output/looker_gap_analysis.csv', index=False)
print(f"✓ Gap analysis exported: {len(gap_summary)} rows")

# Export 4: Host Performance (Pre-aggregated)
host_metrics = analyzer.contacts.groupby('id_host').agg({
    'id_guest': 'count',
    'was_replied': 'mean',
    'was_accepted': 'mean',
    'was_booked': 'mean',
    'response_time_hours': 'median',
    'n_messages': 'mean'
}).reset_index()

host_metrics.columns = ['host_id', 'total_inquiries', 'reply_rate', 'acceptance_rate',
                        'booking_rate', 'median_response_hrs', 'avg_messages']

# Only active hosts
host_metrics = host_metrics[host_metrics['total_inquiries'] >= 3]
host_metrics.to_csv('output/looker_host_metrics.csv', index=False)
print(f"✓ Host metrics exported: {len(host_metrics):,} hosts")

# Export 5: KPI Summary (Single row for scorecards)
kpis = analyzer.calculate_kpis()
kpi_df = pd.DataFrame([{
    'metric': 'Overall Performance',
    'total_searches': kpis['total_searches'],
    'total_inquiries': kpis['total_inquiries'],
    'total_replied': kpis['total_replied'],
    'total_accepted': kpis['total_accepted'],
    'total_booked': kpis['total_booked'],
    'reply_rate': round(kpis['reply_rate'] * 100, 1),
    'acceptance_rate': round(kpis['acceptance_rate'] * 100, 1),
    'booking_rate': round(kpis['booking_rate'] * 100, 1),
    'avg_response_time_hours': round(kpis['avg_response_time_hours'], 1)
}])

kpi_df.to_csv('output/looker_kpis.csv', index=False)
print(f"✓ KPIs exported")

print("\n" + "="*60)
print("✅ ALL FILES EXPORTED TO output/ FOLDER")
print("="*60)
print("\nFiles created:")
print("  1. looker_searches.csv       - Search demand data")
print("  2. looker_contacts.csv       - Booking/inquiry data")
print("  3. looker_gap_analysis.csv   - Supply-demand gaps")
print("  4. looker_host_metrics.csv   - Host performance")
print("  5. looker_kpis.csv           - Key metrics")
print("\nNext: Upload these to Google Drive for Looker Studio!")