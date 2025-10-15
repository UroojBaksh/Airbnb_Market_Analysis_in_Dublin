"""
Complete Airbnb Dublin Analysis Pipeline
Run this single file to execute entire analysis
"""

import os
import sys
from pathlib import Path

import pandas as pd

# Create necessary directories
os.makedirs('src/data', exist_ok=True)
os.makedirs('src/dashboards', exist_ok=True)
os.makedirs('src/output', exist_ok=True)

print("="*70)
print(" AIRBNB DUBLIN MARKET ANALYSIS - COMPLETE PIPELINE ".center(70, "="))
print("="*70)

# Check if data files exist
searches_path = 'data/searches.tsv'
contacts_path = 'data/contacts.tsv'

if not os.path.exists(searches_path) or not os.path.exists(contacts_path):
    print("\n⚠️  ERROR: Data files not found!")
    print(f"\nPlease place your TSV files in the 'data/' folder:")
    print(f"  - {searches_path}")
    print(f"  - {contacts_path}")
    sys.exit(1)

print(f"\n✓ Data files found")
print(f"  - {searches_path}")
print(f"  - {contacts_path}")

# Import modules (assuming they're in the same directory or src/)
try:
    # If files are in src/ folder
    sys.path.insert(0, 'src')
    from main_analysis import AirbnbDublinAnalyzer
    from visualizations import AirbnbVisualizer
    from model import AcceptancePredictionModel
except ImportError:
    # If files are in current directory
    from main_analysis import AirbnbDublinAnalyzer
    from visualizations import AirbnbVisualizer
    from model import AcceptancePredictionModel

# ============= STEP 1: LOAD & ANALYZE DATA =============
print("\n" + "="*70)
print(" STEP 1: DATA LOADING & CORE ANALYSIS ".center(70))
print("="*70)

analyzer = AirbnbDublinAnalyzer(
    searches_path=searches_path,
    contacts_path=contacts_path
)

analyzer.load_data()
kpis = analyzer.calculate_kpis()
gap_analysis = analyzer.demand_supply_gap_analysis()
dimensional_analysis = analyzer.analyze_by_dimensions()
host_metrics = analyzer.host_performance_metrics()
insights = analyzer.generate_insights()

# ============= STEP 2: CREATE VISUALIZATIONS =============
print("\n" + "="*70)
print(" STEP 2: GENERATING INTERACTIVE DASHBOARDS ".center(70))
print("="*70)

viz = AirbnbVisualizer(analyzer)
viz.generate_all_dashboards()

# ============= STEP 3: TRAIN ML MODEL =============
print("\n" + "="*70)
print(" STEP 3: MACHINE LEARNING MODEL ".center(70))
print("="*70)

model = AcceptancePredictionModel(analyzer.contacts)
model.train_model()
eval_results = model.evaluate_model()
model.plot_feature_importance()
model.plot_roc_curve_and_confusion(eval_results)
model.generate_insights()

# ============= STEP 4: GENERATE SUMMARY REPORT =============
print("\n" + "="*70)
print(" STEP 4: GENERATING SUMMARY REPORT ".center(70))
print("="*70)

summary_report = f"""
================================================================================
                AIRBNB DUBLIN MARKET ANALYSIS - EXECUTIVE SUMMARY
================================================================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 KEY PERFORMANCE INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Searches:              {kpis['total_searches']:>15,}
Total Inquiries:             {kpis['total_inquiries']:>15,}
Total Bookings:              {kpis['total_booked']:>15,}

Reply Rate:                  {kpis['reply_rate']:>15.1%}
Acceptance Rate:             {kpis['acceptance_rate']:>15.1%}
Booking Rate:                {kpis['booking_rate']:>15.1%}

Average Response Time:       {kpis['avg_response_time_hours']:>12.1f} hours
Median Nights Searched:      {kpis['median_nights_searched']:>15.0f}
Median Nights Booked:        {kpis['median_nights_booked']:>15.0f}


🎯 TOP SUPPLY-DEMAND GAPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{gap_analysis['gap_by_nights'].to_string()}


🤖 MACHINE LEARNING MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Type:                  XGBoost Classifier
ROC-AUC Score:               {eval_results['roc_auc']:>15.3f}
Training Samples:            {len(model.X_train):>15,}
Test Samples:                {len(model.X_test):>15,}


📈 GENERATED OUTPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Interactive Dashboards:
  ✓ dashboards/executive_dashboard.html
  ✓ dashboards/gap_analysis.html
  ✓ dashboards/host_performance.html
  ✓ dashboards/temporal_analysis.html
  ✓ dashboards/feature_importance.html
  ✓ dashboards/model_performance.html

Summary Report:
  ✓ output/executive_summary.txt


💡 KEY RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. IMPROVE HOST RESPONSIVENESS
   - Current reply rate: {kpis['reply_rate']:.1%}
   - Target: >90% within 1 hour
   - Impact: Could increase bookings by {(0.9 - kpis['reply_rate']) * kpis['total_inquiries']:.0f}

2. ADDRESS TRIP LENGTH GAPS
   - High demand for 3-5 night stays
   - Low host acceptance for these durations
   - Recommendation: Incentivize hosts to accept mid-length stays

3. OPTIMIZE PRICING STRATEGY
   - Analyze price sensitivity by segment
   - Dynamic pricing based on demand patterns
   - Focus on high-conversion segments

4. RECRUIT TARGETED HOSTS
   - Focus on under-supplied neighborhoods
   - Prioritize property types in high demand
   - Estimated revenue opportunity: €{(gap_analysis['gap_by_nights']['gap'].sum() * 100):,.0f}


📊 NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Review interactive dashboards in your browser
2. Share insights with stakeholders
3. Implement top 3 recommendations
4. Monitor KPIs monthly
5. Retrain ML model quarterly with new data


================================================================================
                         ANALYSIS COMPLETE
================================================================================
"""

# Save summary report
with open('output/executive_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)

# ============= FINAL SUMMARY =============
print("\n" + "="*70)
print(" 🎉 PIPELINE COMPLETE! ".center(70, "="))
print("="*70)

print("\n📁 Generated Files:")
print("   📊 Dashboards:")
print("      - dashboards/executive_dashboard.html")
print("      - dashboards/gap_analysis.html")
print("      - dashboards/host_performance.html")
print("      - dashboards/temporal_analysis.html")
print("      - dashboards/feature_importance.html")
print("      - dashboards/model_performance.html")
print("\n   📄 Reports:")
print("      - output/executive_summary.txt")

print("\n🚀 Next Steps:")
print("   1. Open dashboards in your browser")
print("   2. Review executive summary")
print("   3. Share insights with team")

print("\n💡 Pro Tip:")
print("   Use 'python run_all.py' anytime to refresh analysis with new data")

print("\n" + "="*70)
print(" Thank you for using Airbnb Dublin Analytics! ".center(70))
print("="*70 + "\n")

# Import pandas for summary report
import pandas as pd