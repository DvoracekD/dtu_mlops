from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
import pandas as pd
from sklearn import datasets
from evidently.test_suite import TestSuite
from evidently.tests import *

reference_data = datasets.load_iris(as_frame=True).frame
current_data = pd.read_csv('prediction_database.csv')

# Drop the first column from current_data
current_data = current_data.iloc[:, 1:]

# Rename current_data columns with names from reference_data columns
current_data.columns = reference_data.columns

# report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
# report.run(reference_data=reference_data, current_data=current_data)
# report.save_html('report.html')

data_test = TestSuite(tests=[
    TestConflictTarget(),
    TestConflictPrediction(),
    TestTargetPredictionCorrelation(),
    TestHighlyCorrelatedColumns(),
    TestTargetFeaturesCorrelations(),
    TestPredictionFeaturesCorrelations(),
    TestCorrelationChanges(),
])
data_test.run(reference_data=reference_data, current_data=current_data)
data_test.save_html('data_test.html')