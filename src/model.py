"""
XGBoost Model to Predict Inquiry Acceptance
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class AcceptancePredictionModel:
    def __init__(self, contacts_df):
        """Initialize with contacts dataframe"""
        self.contacts = contacts_df.copy()
        self.model = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_features(self):
        """Engineer features for prediction"""
        print("\n" + "=" * 50)
        print("PREPARING FEATURES FOR ML MODEL")
        print("=" * 50)

        df = self.contacts.copy()

        # Remove rows without acceptance data
        df = df.dropna(subset=['ts_contact_at'])

        # Target variable
        df['target_accepted'] = df['ts_accepted_at'].notna().astype(int)

        # Feature engineering
        features = pd.DataFrame()

        # 1. Guest characteristics
        features['n_guests'] = df['n_guests'].fillna(df['n_guests'].median())
        features['n_nights'] = df['n_nights'].fillna(df['n_nights'].median())

        # 2. Communication features
        features['n_messages'] = df['n_messages'].fillna(0)
        features['has_multiple_messages'] = (features['n_messages'] > 1).astype(int)

        # 3. Temporal features
        features['checkin_dow'] = df['checkin_dow'].fillna(0)
        features['contact_dow'] = df['contact_dow'].fillna(0)
        features['checkin_month'] = df['checkin_month'].fillna(0)

        # 4. Lead time
        df['lead_time_days'] = (pd.to_datetime(df['ds_checkin']) - df['ts_contact_at']).dt.days
        features['lead_time_days'] = df['lead_time_days'].fillna(
            df['lead_time_days'].median()
        ).clip(0, 365)


        # 5. Trip characteristics
        features['is_weekend_checkin'] = features['checkin_dow'].isin([5, 6]).astype(int)
        features['is_short_stay'] = (features['n_nights'] <= 2).astype(int)
        features['is_long_stay'] = (features['n_nights'] >= 7).astype(int)
        features['is_large_group'] = (features['n_guests'] >= 4).astype(int)

        # 6. Timing features
        features['is_last_minute'] = (features['lead_time_days'] <= 3).astype(int)
        features['is_far_advance'] = (features['lead_time_days'] >= 60).astype(int)

        # 7. Seasonal features
        features['is_summer'] = features['checkin_month'].isin([6, 7, 8]).astype(int)
        features['is_holiday_season'] = features['checkin_month'].isin([12, 1]).astype(int)

        # Target
        target = df['target_accepted']

        # Remove any remaining NaNs
        features = features.fillna(0)

        self.feature_names = features.columns.tolist()

        print(f"✓ Features prepared: {len(self.feature_names)} features")
        print(f"✓ Dataset size: {len(features):,} inquiries")
        print(f"✓ Acceptance rate: {target.mean():.2%}")
        print(f"\nFeatures: {', '.join(self.feature_names)}")

        return features, target

    def train_model(self, test_size=0.2, random_state=42):
        """Train XGBoost model"""
        print("\n" + "=" * 50)
        print("TRAINING XGBOOST MODEL")
        print("=" * 50)

        # Prepare features
        X, y = self.prepare_features()

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nTrain set: {len(self.X_train):,} samples")
        print(f"Test set: {len(self.X_test):,} samples")

        # Initialize model
        self.model = XGBClassifier(
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )

        # Train
        print("\nTraining model...")
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train,
            cv=5, scoring='roc_auc'
        )

        print(f"\n✓ Model trained successfully!")
        print(f"✓ Cross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        return self.model

    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        print(f"\n🎯 ROC-AUC Score: {roc_auc:.3f}")
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred,
                                    target_names=['Rejected', 'Accepted']))

        print("\n📉 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"True Negatives: {cm[0, 0]:,} | False Positives: {cm[0, 1]:,}")
        print(f"False Negatives: {cm[1, 0]:,} | True Positives: {cm[1, 1]:,}")

        return {
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }

    def plot_feature_importance(self):
        """Visualize feature importance"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig = px.bar(
            importance_df.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title='<b>Top 15 Features Predicting Inquiry Acceptance</b>',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )

        fig.write_html('dashboards/feature_importance.html')
        print("\n✓ Feature importance plot saved")

        return fig

    def plot_roc_curve_and_confusion(self, eval_results):
        """Create ROC curve and confusion matrix visualizations"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('ROC Curve', 'Confusion Matrix'),
            specs=[[{'type': 'scatter'}, {'type': 'heatmap'}]]
        )

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, eval_results['probabilities'])

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC (AUC = {eval_results["roc_auc"]:.3f})',
                line=dict(color='#2ecc71', width=3)
            ),
            row=1, col=1
        )

        # Diagonal reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', dash='dash')
            ),
            row=1, col=1
        )

        # Confusion Matrix
        cm = eval_results['confusion_matrix']

        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted Rejected', 'Predicted Accepted'],
                y=['Actually Rejected', 'Actually Accepted'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues',
                showscale=False
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)

        fig.update_layout(
            height=500,
            title_text="<b>Model Performance Metrics</b>",
            showlegend=True
        )

        fig.write_html('dashboards/model_performance.html')
        print("✓ Model performance plots saved")

        return fig

    def predict_acceptance_probability(self, inquiry_data):
        """Predict acceptance probability for new inquiry"""
        # inquiry_data should be a dict or DataFrame with required features
        if isinstance(inquiry_data, dict):
            inquiry_df = pd.DataFrame([inquiry_data])
        else:
            inquiry_df = inquiry_data

        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in inquiry_df.columns:
                inquiry_df[feature] = 0

        # Predict
        proba = self.model.predict_proba(inquiry_df[self.feature_names])[:, 1]

        return proba[0]

    def generate_insights(self):
        """Generate business insights from model"""
        print("\n" + "=" * 50)
        print("🎯 MODEL INSIGHTS & RECOMMENDATIONS")
        print("=" * 50)

        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n📊 TOP 5 FACTORS PREDICTING ACCEPTANCE:")
        for i, row in importance_df.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")

        # Analyze impact of key features
        test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        test_df['actual'] = self.y_test.values
        test_df['predicted_proba'] = self.model.predict_proba(self.X_test)[:, 1]

        # Message impact
        low_messages = test_df[test_df['n_messages'] <= 1]['predicted_proba'].mean()
        high_messages = test_df[test_df['n_messages'] > 3]['predicted_proba'].mean()

        print(f"\n💬 MESSAGE COUNT IMPACT:")
        print(f"   Low messages (≤1): {low_messages:.1%} avg acceptance probability")
        print(f"   High messages (>3): {high_messages:.1%} avg acceptance probability")
        print(f"   Impact: {(high_messages - low_messages) * 100:.1f}pp increase")

        # Lead time impact
        last_minute = test_df[test_df['is_last_minute'] == 1]['predicted_proba'].mean()
        planned = test_df[test_df['is_last_minute'] == 0]['predicted_proba'].mean()

        print(f"\n⏰ BOOKING TIMING IMPACT:")
        print(f"   Last minute (≤3 days): {last_minute:.1%} avg acceptance")
        print(f"   Planned ahead: {planned:.1%} avg acceptance")

        # Trip length impact
        short_stay = test_df[test_df['is_short_stay'] == 1]['predicted_proba'].mean()
        long_stay = test_df[test_df['is_long_stay'] == 1]['predicted_proba'].mean()

        print(f"\n📅 TRIP LENGTH IMPACT:")
        print(f"   Short stays (≤2 nights): {short_stay:.1%} avg acceptance")
        print(f"   Long stays (≥7 nights): {long_stay:.1%} avg acceptance")

        print("\n💡 RECOMMENDATIONS:")
        print("   1. Encourage guests to send more messages (builds rapport)")
        print("   2. Promote advance bookings over last-minute requests")
        print("   3. Optimize pricing/availability for high-probability inquiries")
        print("   4. Train hosts on factors that increase acceptance likelihood")


# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    from main_analysis import AirbnbDublinAnalyzer

    # Load data
    print("Loading data...")
    analyzer = AirbnbDublinAnalyzer(
        searches_path='data/searches.tsv',
        contacts_path='data/contacts.tsv'
    )
    analyzer.load_data()

    # Initialize and train model
    model = AcceptancePredictionModel(analyzer.contacts)
    model.train_model()

    # Evaluate
    eval_results = model.evaluate_model()

    # Visualize
    model.plot_feature_importance()
    model.plot_roc_curve_and_confusion(eval_results)

    # Insights
    model.generate_insights()

    print("\n✅ Model training and evaluation complete!")

    # Example prediction
    print("\n" + "=" * 50)
    print("EXAMPLE PREDICTION")
    print("=" * 50)

    sample_inquiry = {
        'n_guests': 2,
        'n_nights': 3,
        'n_messages': 2,
        'checkin_dow': 5,  # Friday
        'contact_dow': 2,  # Tuesday
        'checkin_month': 7,  # July
        'lead_time_days': 14,
        'has_multiple_messages': 1,
        'is_weekend_checkin': 1,
        'is_short_stay': 0,
        'is_long_stay': 0,
        'is_large_group': 0,
        'is_last_minute': 0,
        'is_far_advance': 0,
        'is_summer': 1,
        'is_holiday_season': 0
    }

    probability = model.predict_acceptance_probability(sample_inquiry)
    print(f"\n📩 Sample Inquiry Characteristics:")
    print(f"   - 2 guests, 3 nights")
    print(f"   - Friday check-in, 2 weeks advance")
    print(f"   - July (summer), 2 messages exchanged")
    print(f"\n🎯 Predicted Acceptance Probability: {probability:.1%}")