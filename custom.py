import math
import pytz
from datetime import datetime
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.terminal_theme import MONOKAI
import robusta_krr
import numpy as np
import pydantic as pd
from robusta_krr.api import formatters
from robusta_krr.api.models import Result
from robusta_krr.core.abstract.strategies import (
    BaseStrategy,
    K8sObjectData,
    MetricsPodData,
    PodsTimeData,
    ResourceRecommendation,
    ResourceType,
    RunResult,
    StrategySettings,
)
from robusta_krr.core.integrations.prometheus.metrics import (
    MaxMemoryLoader,
    PercentileCPULoader,
    PrometheusMetric,
    CPUAmountLoader,
    MemoryAmountLoader,
)
from robusta_krr.formatters import json, table
from robusta_krr.utils.resource_units import parse


SharedResourceWorkload = [
    "judger",
    "multiple-file-judger",
    "mysql-judger",
    "postgresql-judger",
    "sqlserver-judger",
]


def PercentileMemoryLoader(percentile: float) -> type[PrometheusMetric]:
    """
    A factory for creating percentile CPU usage metric loaders.
    """

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    class PercentileMemoryLoader(PrometheusMetric):
        def get_query(self, object: K8sObjectData, duration: str, step: str) -> str:
            pods_selector = "|".join(pod.name for pod in object.pods)
            cluster_label = self.get_prometheus_cluster_label()

            return f"""
                quantile_over_time(
                    {round(percentile / 100, 2)},
                    max(
                        container_memory_working_set_bytes{{
                            namespace="{object.namespace}",
                            pod=~"{pods_selector}",
                            container="{object.container}"
                            {cluster_label}
                        }}
                    ) by (container, pod, job)
                    [{duration}:{step}]
                )
            """

    return PercentileMemoryLoader


def PercentileCPULoader2(percentile: float) -> type[PrometheusMetric]:
    """
    A factory for creating percentile CPU usage metric loaders.
    """

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    class PercentileCPULoader2(PrometheusMetric):
        def get_query(self, object: K8sObjectData, duration: str, step: str) -> str:
            pods_selector = "|".join(pod.name for pod in object.pods)
            cluster_label = self.get_prometheus_cluster_label()
            return f"""
                quantile_over_time(
                    {round(percentile / 100, 2)},
                    max(
                        rate(
                            container_cpu_usage_seconds_total{{
                                namespace="{object.namespace}",
                                pod=~"{pods_selector}",
                                container="{object.container}"
                                {cluster_label}
                            }}[{step}]
                        )
                    ) by (container, pod, job)
                    [{duration}:{step}]
                )
            """

    return PercentileCPULoader2


@formatters.register(display_name="custom", rich_console=True)
def custom_formatter(result: Result) -> str:
    Path("static").mkdir(parents=True, exist_ok=True)

    with open("static/result.json", "w") as f:
        result.format(json)
        f.write(result.format(json))

    now = datetime.now(pytz.timezone("Asia/Shanghai"))
    console = Console(width=192, record=True)
    table_result = result.format(table)
    for c in table_result.columns:
        c.overflow = "fold"
    console.print(table_result)
    console.save_svg(
        "static/result.svg",
        title="KRR result @ %s" % now.strftime("%d-%m-%Y %H:%M:%S %Z"),
        font_aspect_ratio=0.61,
        theme=MONOKAI,
    )

    return "json and table result generated."


class CustomStrategySettings(StrategySettings):
    cpu_percentile: float = pd.Field(
        99,
        gt=0,
        le=100,
        description="The percentile to use for the CPU recommendation.",
    )
    cpu_percentile_for_shared: float = pd.Field(
        50,
        gt=0,
        le=100,
        description="The percentile to use for the CPU recommendation.",
    )
    memory_percentile_for_shared: float = pd.Field(
        50, gt=0, le=100, description="The percentile to use for Memory recommendation."
    )
    memory_buffer_percentage_for_large: float = pd.Field(
        5,
        gt=0,
        description="The percentage of added buffer to the peak memory usage for memory recommendation. (3Gi, +Inf]",
    )
    memory_buffer_percentage_for_medium: float = pd.Field(
        10,
        gt=0,
        description="The percentage of added buffer to the peak memory usage for memory recommendation. (1Gi, 3Gi]",
    )
    memory_buffer_percentage_for_small: float = pd.Field(
        15,
        gt=0,
        description="The percentage of added buffer to the peak memory usage for memory recommendation. (0, 1Gi]",
    )
    points_required: int = pd.Field(
        1000,
        ge=1,
        description="The number of data points required to make a recommendation for a resource.",
    )
    points_required_for_jobs: int = pd.Field(
        5,
        ge=1,
        description="The number of data points required to make a recommendation for a resource.",
    )

    def calculate_memory_proposal(self, data: PodsTimeData, avg: bool = False) -> float:
        if not avg:
            data_ = [np.max(values[:, 1]) for values in data.values()]
        else:
            data_ = [np.average(values[:, 1]) for values in data.values()]
        if len(data_) == 0:
            return float("NaN")

        raw = np.max(data_)
        memory_buffer_percentage = (
            self.memory_buffer_percentage_for_large
            if (raw > parse("3Gi"))
            else (
                self.memory_buffer_percentage_for_medium
                if (raw > parse("1Gi"))
                else self.memory_buffer_percentage_for_small
            )
        )
        raw *= 1 + memory_buffer_percentage / 100
        if raw < parse("64Mi"):
            round_base = parse("16Mi")
        elif raw < parse("256Mi"):
            round_base = parse("64Mi")
        else:
            round_base = parse("256Mi")
        return round(math.ceil(raw / round_base) * round_base, 10)

    def calculate_cpu_proposal(self, data: PodsTimeData) -> float:
        if len(data) == 0:
            return float("NaN")

        if len(data) > 1:
            data_ = np.concatenate([values[:, 1] for values in data.values()])
        else:
            data_ = list(data.values())[0][:, 1]

        raw = np.max(data_)
        if raw < parse("100m"):
            round_base = parse("10m")
        else:
            round_base = parse("100m")
        return round(math.ceil(raw / round_base) * round_base, 10)

    def get_original_memory(
        self, data: K8sObjectData, proposal: Optional[float]
    ) -> Optional[float]:
        value = data.allocations.limits[ResourceType.Memory]
        if proposal is not None and (value is None or value < proposal):
            return proposal
        if value is None:
            return None
        if value == "?":
            return float("NaN")
        return value


class CustomStrategy(BaseStrategy[CustomStrategySettings]):
    """
    CPU request: {cpu_percentile}% percentile, limit: original value
    CPU request: {cpu_percentile_for_shared}% percentile, limit: original value, for specific workload (ie. judger)
    Memory request: max + {memory_buffer_percentage_for_large}%, limit: original value, (3Gi, +Inf]
    Memory request: max + {memory_buffer_percentage_for_medium}%, limit: original value, (1Gi, 3Gi]
    Memory request: max + {memory_buffer_percentage_for_small}%, limit: original value, (0, 1Gi]
    Memory request: {memory_percentile_for_shared}% percentile, for specific workload (ie. judger)...
    History: {history_duration} hours
    Step: {timeframe_duration} minutes
    Minimum data points: {points_required} for pods (Dataset size less than it will be ignored)
    Minimum data points: {points_required_for_jobs} for jobs (Dataset size less than it will be ignored)

    This strategy does not work with objects with HPA defined (Horizontal Pod Autoscaler).
    If HPA is defined for CPU or Memory, the strategy will return "?" for that resource.

    Learn more: [underline]https://github.com/robusta-dev/krr#algorithm[/underline]

    For Avoiding tiny jitter making result unstable.
    CPU will be rounding by 10m if less than 100m, otherwise 100m
    Memroy will be rounding by 16Mi if less than 64Mi, by 64Mi if less than 256Mi, by 256Mi otherwise
    """

    display_name = "custom"
    rich_console = True

    @property
    def metrics(self) -> list[type[PrometheusMetric]]:
        return [
            PercentileCPULoader(self.settings.cpu_percentile),
            PercentileCPULoader2(self.settings.cpu_percentile_for_shared),
            MaxMemoryLoader,
            PercentileMemoryLoader(self.settings.memory_percentile_for_shared),
            CPUAmountLoader,
            MemoryAmountLoader,
        ]

    def __calculate_cpu_proposal(
        self, history_data: MetricsPodData, object_data: K8sObjectData
    ) -> ResourceRecommendation:
        if object_data.name not in SharedResourceWorkload:
            data = history_data["PercentileCPULoader"]
        else:
            data = history_data["PercentileCPULoader2"]

        if len(data) == 0:
            return ResourceRecommendation.undefined(info="No data")

        data_count = {
            pod: values[0, 1] for pod, values in history_data["CPUAmountLoader"].items()
        }
        # Here we filter out pods from calculation that have less than `points_required` data points
        if object_data.kind == "Job":
            points_required = self.settings.points_required_for_jobs
        else:
            points_required = self.settings.points_required
        filtered_data = {
            pod: values
            for pod, values in data.items()
            if data_count.get(pod, 0) >= points_required
        }

        if len(filtered_data) == 0:
            return ResourceRecommendation.undefined(info="Not enough data")

        if (
            object_data.hpa is not None
            and object_data.hpa.target_cpu_utilization_percentage is not None
        ):
            return ResourceRecommendation.undefined(info="HPA detected")

        cpu_request = self.settings.calculate_cpu_proposal(filtered_data)
        # always return None: https://home.robusta.dev/blog/stop-using-cpu-limits
        return ResourceRecommendation(request=cpu_request, limit=None, info=None)

    def __calculate_memory_proposal(
        self, history_data: MetricsPodData, object_data: K8sObjectData
    ) -> ResourceRecommendation:
        if object_data.name not in SharedResourceWorkload:
            data = history_data["MaxMemoryLoader"]
            avg = False
        else:
            data = history_data["PercentileMemoryLoader"]
            avg = True

        if len(data) == 0:
            return ResourceRecommendation.undefined(info="No data")

        data_count = {
            pod: values[0, 1]
            for pod, values in history_data["MemoryAmountLoader"].items()
        }
        # Here we filter out pods from calculation that have less than `points_required` data points
        if object_data.kind == "Job":
            points_required = self.settings.points_required_for_jobs
        else:
            points_required = self.settings.points_required
        filtered_data = {
            pod: value
            for pod, value in data.items()
            if data_count.get(pod, 0) >= points_required
        }

        if len(filtered_data) == 0:
            return ResourceRecommendation.undefined(info="Not enough data")

        if (
            object_data.hpa is not None
            and object_data.hpa.target_memory_utilization_percentage is not None
        ):
            return ResourceRecommendation.undefined(info="HPA detected")

        memory_request = self.settings.calculate_memory_proposal(filtered_data, avg)
        memory_limit = self.settings.get_original_memory(object_data, memory_request)
        return ResourceRecommendation(
            request=memory_request, limit=memory_limit, info=None
        )

    def run(
        self, history_data: MetricsPodData, object_data: K8sObjectData
    ) -> RunResult:
        return {
            ResourceType.CPU: self.__calculate_cpu_proposal(history_data, object_data),
            ResourceType.Memory: self.__calculate_memory_proposal(
                history_data, object_data
            ),
        }


# Running this file will register the formatter and make it available to the CLI
# Run it as `python ./custom.py simple --formater my_formatter`
if __name__ == "__main__":
    robusta_krr.run()
