"""Unit tests for DeepAnalyze Dynamic Charting & Visual Analytics Engine."""

import unittest
import polars as pl
from deepanalyze.charts import (
    detect_dimensions_and_measures,
    render_histogram,
    render_box_plot,
    render_correlation_heatmap,
    render_categorical_bar,
    render_time_trend,
    render_scatter_density,
    render_regional_density,
    render_missingness_waterfall,
    render_pareto_analysis,
    render_quantile_qq,
    render_crosstab_heatmap,
    export_chart_script,
)


class TestChartsEngine(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({
            "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "customer": ["Alpha", "Beta", "Gamma", "Delta", "Alpha", "Beta", "Alpha", "Gamma", "Delta", "Alpha"],
            "region": ["Central", "Western", "Eastern", "Central", "Western", "Eastern", "Central", "Western", "Eastern", "Central"],
            "category": ["Tech", "Tech", "Office", "Furniture", "Tech", "Office", "Tech", "Furniture", "Tech", "Office"],
            "sales": [1200.0, 450.0, 890.0, 3100.0, 750.0, 210.0, 1800.0, 950.0, 1400.0, 500.0],
            "quantity": [2, 1, 3, 5, 1, 2, 4, 2, 3, 1],
            "profit": [240.0, 90.0, 178.0, 620.0, 150.0, 42.0, 360.0, 190.0, 280.0, 100.0],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                     "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10"]
        })

    def test_dimensions_and_measures_detection(self):
        dims, measures = detect_dimensions_and_measures(self.df)
        self.assertIn("customer", dims)
        self.assertIn("region", dims)
        self.assertIn("category", dims)
        self.assertIn("sales", measures)
        self.assertIn("profit", measures)

    def test_histogram_rendering(self):
        chart, stats = render_histogram(self.df, "sales", bins=5)
        self.assertIn("DISTRIBUTION HISTOGRAM", chart)
        self.assertEqual(stats["n"], 10)
        self.assertGreater(stats["max"], stats["min"])

    def test_box_plot_rendering(self):
        chart, stats = render_box_plot(self.df, "sales")
        self.assertIn("BOX & WHISKER", chart)
        self.assertIn("sales", stats)
        self.assertIn("q1", stats["sales"])
        self.assertIn("median", stats["sales"])
        self.assertIn("q3", stats["sales"])

    def test_correlation_heatmap(self):
        chart, stats = render_correlation_heatmap(self.df)
        self.assertIn("CORRELATION", chart)
        self.assertIn("matrix", stats)

    def test_categorical_bar(self):
        chart, stats = render_categorical_bar(self.df, "category", measure="sales", agg="SUM")
        self.assertIn("CATEGORICAL", chart)
        self.assertIn("Tech", stats["categories"])

    def test_time_trend(self):
        chart, stats = render_time_trend(self.df, "date", "sales", agg="SUM")
        self.assertIn("CHRONOLOGICAL TREND", chart)
        self.assertEqual(len(stats["dates"]), 10)

    def test_scatter_density(self):
        chart, stats = render_scatter_density(self.df, "sales", "profit")
        self.assertIn("SCATTER DENSITY MAP", chart)
        self.assertEqual(stats["n"], 10)

    def test_regional_density(self):
        chart, stats = render_regional_density(self.df, "region")
        self.assertIn("GEOGRAPHICAL / REGIONAL DENSITY", chart)
        self.assertEqual(stats["total"], 10)

    def test_missingness_waterfall(self):
        chart, stats = render_missingness_waterfall(self.df)
        self.assertIn("DATA QUALITY & COMPLETENESS WATERFALL", chart)
        self.assertEqual(len(stats["columns"]), len(self.df.columns))

    def test_pareto_analysis(self):
        chart, stats = render_pareto_analysis(self.df, "category", measure_col="sales")
        self.assertIn("PARETO DISTRIBUTION (80/20 ANALYSIS)", chart)
        self.assertIn("cum_percentages", stats)

    def test_quantile_qq(self):
        chart, stats = render_quantile_qq(self.df, "sales")
        self.assertIn("QUANTILE & EMPIRICAL NORMALITY LADDER", chart)
        self.assertIn("iqr", stats)
        self.assertEqual(len(stats["percentiles"]), 9)

    def test_crosstab_heatmap(self):
        chart, stats = render_crosstab_heatmap(self.df, "category", "region")
        self.assertIn("2D CROSS-TABULATION MATRIX", chart)
        self.assertIn("matrix", stats)

    def test_export_chart_script_plotly_and_matplotlib(self):
        plotly_code = export_chart_script("histogram", "sales", output_format="plotly")
        self.assertIn("import plotly.express as px", plotly_code)
        self.assertIn("px.histogram", plotly_code)

        mpl_code = export_chart_script("histogram", "sales", output_format="matplotlib")
        self.assertIn("import matplotlib.pyplot as plt", mpl_code)
        self.assertIn("plt.hist", mpl_code)


if __name__ == "__main__":
    unittest.main()
