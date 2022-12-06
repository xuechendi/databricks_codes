from typing import Any, Dict, List, Optional, Set, Tuple, Iterable, Union

from databricks.automl.shared.const import Metric, ProblemType
from databricks.automl.shared.result import TrialInfo


class CellOutput:
    SHOWING_IN_SCIENTIFIC_NOTATION_THRESHOLD_UPPER = 1e+6
    SHOWING_IN_SCIENTIFIC_NOTATION_THRESHOLD_LOWER = 1e-6

    @staticmethod
    def get_summary_html(trial: TrialInfo,
                         data_exp_url: str,
                         experiment_url: str,
                         metric: Metric,
                         problem_type: ProblemType,
                         sample_fraction: Optional[float] = None) -> str:

        metric_table_html = CellOutput._get_metric_table(problem_type, metric, trial)
        mlflow_exp_link = CellOutput._get_link_html(experiment_url, "MLflow experiment")
        data_exp_link = CellOutput._get_link_html(data_exp_url, "data exploration notebook")
        data_exp_div = f"<div><p>For exploratory data analysis, open the {data_exp_link}</p></div>"
        best_trial_nb_link = CellOutput._get_link_html(trial.notebook_url, "best trial notebook")

        sampling_div = ""
        # NOTE: We don't do any sampling for ProblemType.FORECAST
        if sample_fraction and problem_type in {ProblemType.CLASSIFICATION, ProblemType.REGRESSION}:
            pct = sample_fraction * 100
            sampling_type = "stratified" if problem_type == ProblemType.CLASSIFICATION else "simple random"

            sampling_div = "<div><p>Data exploration and trials were run on a <strong>{:.3f}%</strong> sample of the usable rows in the dataset. This dataset was sampled using {} sampling.</p></div>".format(
                pct, sampling_type)

        return f"""
        <style>
            .grid-container {{
              display: grid
              grid-template-columns: auto;
              padding: 10px;
            }}
            <!-- Picked to be same as https://github.com/databricks/universe/blob/feaafc3875d9b95a124ed44ff4b99fb1002e544d/webapp/web/js/templates/iframeSandbox.css#L6-L11 -->
            .grid-container div {{
              font-family: Helvetica, Arial, sans-serif;
              font-size: 14px;
            }}
        </style>
        <div class="grid-container">
            {sampling_div}
            {data_exp_div}
            <div><p>To view the best performing model, open the {best_trial_nb_link}</p></div>
            <div><p>To view details about all trials, navigate to the {mlflow_exp_link}</p></div>
            <div><p><strong>Metrics for the best trial:</strong></p></div>
            <div>
                <!-- class inlined from https://github.com/databricks/universe/blob/feaafc3875d9b95a124ed44ff4b99fb1002e544d/webapp/web/js/templates/iframeSandbox.css#L35 -->
                {metric_table_html}
            </div>
        </div>
        """

    @staticmethod
    def _get_link_html(url: str, text: str) -> str:
        return f"<a href={url}>{text}</a>"

    @staticmethod
    def _get_metric_table(problem_type: ProblemType, metric: Metric, trial: TrialInfo) -> str:
        metrics_to_display = CellOutput._get_metrics_to_display(problem_type, metric, trial.metrics)

        if problem_type == ProblemType.FORECAST:
            formatted_rows = [
                f"""
                <tr>
                    <th> {metric_name} </th>
                    <td> {tst} </td>
                </tr>
                """ for metric_name, tst in metrics_to_display
            ]
            rows = "\n".join(formatted_rows)

            return f"""
                    <table class="dataframe">
                        <thead>
                          <tr>
                            <th></th>
                            <th>Validation</th>
                          </tr>
                        </thead>
                        <tbody>
                        {rows}
                        </tbody>
                    </table>
            """
        else:
            formatted_rows = [
                f"""
                <tr>
                    <th> {metric_name} </th>
                    <td> {trn} </td>
                    <td> {val} </td>
                    <td> {tst} </td>
                </tr>
                """ for metric_name, (trn, val, tst) in metrics_to_display
            ]
            rows = "\n".join(formatted_rows)

            return f"""
                    <table class="dataframe">
                        <thead>
                          <tr>
                            <th></th>
                            <th>Train</th>
                            <th>Validation</th>
                            <th>Test</th>
                          </tr>
                        </thead>
                        <tbody>
                        {rows}
                        </tbody>
                    </table>
            """

    @staticmethod
    def _get_metrics_to_display(problem_type: ProblemType, metric: Metric,
                                metrics: Dict[str, float]) -> List[Tuple[str, Tuple]]:
        """
        Returns a dictionary with the key as metric name (without prefix) and value as a tuple of
        strings with (validation, train) metric values for the metric name.
        We also round the metric to 3 decimal points and return None if any of the metrics aren't logged
        :param metrics: metric dictionary logged into MLflow
        :return: 
            Classification/regression returns List[Tuple[str, Tuple[str, str, str]]]: {metric_name: (train_metrics, val_metrics, test_metrics)}
            Forecasting returns List[Tuple[str, str]]: {metric_name: val_metric}
        """
        if problem_type == ProblemType.FORECAST:
            val_metrics = {}

            for metric_name, value in metrics.items():
                val_metrics[metric_name.replace("val_", "")] = CellOutput._get_display_format(value)

            display_metrics = {}
            for metric_name in set(val_metrics.keys()):
                val = val_metrics.get(metric_name, "None")
                display_metrics[metric_name] = val

            # re-arrange the metrics to display the primary metric at the top
            metric = metric.trial_metric_name.replace("val_", "")
            if metric in display_metrics.keys():
                primary_metric = (metric, display_metrics[metric])
                del display_metrics[metric]

                display_metrics = [(k, v) for k, v in display_metrics.items()]
                return [primary_metric] + display_metrics
            return [(k, v) for k, v in display_metrics.items()]
        else:
            train_metrics = {}
            val_metric = {}
            test_metric = {}

            for metric_name, value in metrics.items():
                if metric_name.startswith("training_"):
                    train_metrics[metric_name.replace("training_",
                                                      "")] = CellOutput._get_display_format(value)
                elif metric_name.startswith("val_"):
                    val_metric[metric_name.replace("val_",
                                                   "")] = CellOutput._get_display_format(value)
                elif metric_name.startswith("test_"):
                    test_metric[metric_name.replace("test_",
                                                    "")] = CellOutput._get_display_format(value)

            display_metrics = {}
            for metric_name in set(train_metrics.keys()).union(set(val_metric.keys())).union(
                    set(test_metric.keys())):
                train = train_metrics.get(metric_name, "None")
                val = val_metric.get(metric_name, "None")
                test = test_metric.get(metric_name, "None")
                display_metrics[metric_name] = (train, val, test)

            # re-arrange the metrics to display the primary metric at the top
            metric = metric.trial_metric_name.replace("val_", "")
            if metric in display_metrics.keys():
                primary_metric = (metric, display_metrics[metric])
                del display_metrics[metric]

                display_metrics = [(k, v) for k, v in display_metrics.items()]
                return [primary_metric] + display_metrics
            return [(k, v) for k, v in display_metrics.items()]

    @staticmethod
    def _get_display_format(num: float) -> str:
        abs_value = abs(num)
        if (abs_value > CellOutput.SHOWING_IN_SCIENTIFIC_NOTATION_THRESHOLD_UPPER) \
                or (abs_value < CellOutput.SHOWING_IN_SCIENTIFIC_NOTATION_THRESHOLD_LOWER):
            return f"{num:.6e}"
        else:
            return f"{num:.3f}"
